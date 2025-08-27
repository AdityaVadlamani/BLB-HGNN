import glob
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime

import lightning as L
import torch
import wandb
from constants import MODEL_OUTPUT_PATH
from data_module import DataModule
from lightning_utilities.core.rank_zero import rank_zero_info
from multibisage_module import MultiBiSageModule
from setup_utils import setup_ensemble_objects, setup_replica_objects


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve"
    )
    parser.add_argument(
        "--dataset",
        default="OGB_MAG",
        choices=[
            "OGB_MAG",
            "academic4HetGNN",
            "ohgbn-yelp2",
            "ohgbn-Freebase",
            "IMDB",
            "DBLP",
        ],
    )

    parser.add_argument("--seed", default=0, type=int, help="random seed")

    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="number of epochs for training (r parameter)",
    )

    parser.add_argument(
        "--ensemble_epochs",
        default=50,
        type=int,
        help="number of epochs for fine-tuning averaged model",
    )

    parser.add_argument(
        "--batch_size", default=128, type=int, help="batch size for training"
    )

    parser.add_argument(
        "--num_neighbors", default=50, type=int, help="number of neighbors"
    )

    parser.add_argument(
        "--embedding_dim", default=512, type=int, help="embedding dimension"
    )

    parser.add_argument(
        "--num_heads",
        default=8,
        type=int,
        help="number of heads for transformer encoders",
    )

    parser.add_argument(
        "--num_layers",
        default=4,
        type=int,
        help="number of layers for transformer encoders",
    )

    parser.add_argument(
        "--dropout",
        default=0.5,
        type=float,
        help="dropout",
    )

    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="number of workers for DataLoader",
    )

    parser.add_argument(
        "--n_sample_multiplier",
        default=1,
        type=float,
        help="multiplier to determine max amount of training data to utilize (b parameter) if random sampling is used",
    )

    parser.add_argument(
        "--replica_set_frac",
        default=0.75,
        type=float,
        help="multiplier to determine subset size for bootstrap sampler (b_rep parameter)",
    )

    parser.add_argument(
        "--num_subsets",
        default=2,
        type=int,
        help="number of subsets to boostrap from (s parameter)",
    )

    parser.add_argument(
        "--sampler_type",
        type=str,
        default="uniform",
        # choices=["uniform", "ppr", "ppr_weighted", "ss", "ppr_ss", "ppr_ss_weighted", "ppr_simple"],
        help="sampler type",
    )

    parser.add_argument(
        "--ensemble_method",
        default="avg_params_finetune",
        choices=["avg_params", "avg_params_finetune", "voting", "deep_ensemble"],
        help="ensemble method",
    )

    parser.add_argument(
        "--no_bootstrap",
        action="store_true",
        help="run baseline with no bag of little bootstraps",
    )

    parser.add_argument(
        "--partition_across_replicas",
        action="store_true",
        help="whether to sample or use partition for model replica data",
    )

    parser.add_argument(
        "--no_ensemble",
        action="store_true",
        help="no ensembling/model averaging",
    )

    parser.add_argument(
        "--skip_replicas",
        action="store_true",
        help="skip training replicas and go straight to ensemble model",
    )

    parser.add_argument(
        "--use_dynamic_replica_set_frac",
        action="store_true",
        help="don't use replica_set_frac if provided but instead scale it based on num_subsets",
    )

    parser.add_argument(
        "--start_replica", default=0, type=int, help="replica id to start with"
    )
    parser.add_argument(
        "--end_replica", default=-1, type=int, help="replica id to end with (inclusive)"
    )

    parser.add_argument(
        "--replicas_to_average",
        default=[],
        type=int,
        nargs="+",
        help="replicas to consider for averaging (default is empty list which is all)",
    )
    parser.add_argument(
        "--run_only_test",
        action="store_true",
        help="Run only trainer.test",
    )

    # SLURM-related arguments
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="number of SLURM nodes",
    )

    parser.add_argument(
        "--num_gpus",
        default=-1,
        type=int,
        help="number of GPUs per SLURM node",
    )

    parser.add_argument(
        "--job_id",
        default=datetime.now().strftime("%d-%m-%Y-%H:%M:%S"),
        type=str,
        help="SLURM job id or current date if not provided",
    )

    parser.add_argument(
        "--resume_from",
        default=[""],
        type=str,
        nargs="+",
        help="List of SLURM job IDs to resume training (from earliest to latest)",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    torch.set_float32_matmul_precision("high")

    args = parse_args()
    
    if not os.path.exists(MODEL_OUTPUT_PATH):
        os.makedirs(MODEL_OUTPUT_PATH)

    L.seed_everything(args.seed, workers=True)

    # TODO: See if no overlap can be used for ppr and ss
    if args.partition_across_replicas and args.sampler_type == "uniform":
        args.use_dynamic_replica_set_frac = True

    if args.no_bootstrap:
        rank_zero_info(f"Change num_subsets from {args.num_subsets} to 1")
        args.num_subsets = 1
        args.start_replica = 0
        args.end_replica = 0

    if args.use_dynamic_replica_set_frac:
        args.replica_set_frac = 1 / args.num_subsets

    if args.end_replica == -1:
        args.end_replica = args.num_subsets - 1

    rank_zero_info(f"Args: {args}")
    rank_zero_info("Setting up data module")
    datamodule = DataModule(
        name=args.dataset,
        seed=args.seed,
        n_sample_multiplier=args.n_sample_multiplier,
        replica_set_frac=args.replica_set_frac,
        num_subsets=args.num_subsets,
        sampler_type=args.sampler_type,
        partition_across_replicas=args.partition_across_replicas,
        world_size=args.num_gpus * args.num_nodes,
        num_neighs=args.num_neighbors,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    datamodule.set_train_subset()
    rank_zero_info("Finished setting up data module")

    run_name = None
    if not args.skip_replicas:
        for i in range(args.start_replica, args.end_replica + 1):
            L.seed_everything(args.seed, workers=True)
            model, trainer, best_callback, run_name = setup_replica_objects(
                args, datamodule, i, run_name
            )
            resume_ckpt_path = (
                f"snapshot_{args.dataset}_{args.resume_from[-1]}_{i}.ckpt"
            )
            if not args.resume_from[-1]:
                resume_ckpt_path = None
            elif not os.path.exists(resume_ckpt_path):
                resume_ckpt_path = glob.glob(
                    os.path.join(
                        MODEL_OUTPUT_PATH,
                        args.dataset,
                        f"{args.resume_from[-1]}_*_{i}.ckpt",
                    )
                )
                if not resume_ckpt_path or not os.path.exists(resume_ckpt_path[0]):
                    resume_ckpt_path = None
                else:
                    resume_ckpt_path = resume_ckpt_path[0]

            if args.no_bootstrap:
                datamodule.set_replica()
            else:
                datamodule.set_replica(i)

            if not args.run_only_test:
                trainer.fit(
                    model=model,
                    datamodule=datamodule,
                    ckpt_path=resume_ckpt_path,
                )
                resume_ckpt_path = best_callback.best_model_path
            trainer.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=resume_ckpt_path,
            )
            wandb.finish()

    if args.no_ensemble:
        return

    # ENSEMBLE MODEL
    if args.replicas_to_average:
        args.replicas_to_average = list(
            filter(lambda x: 0 <= x < args.num_subsets, args.replicas_to_average)
        )
    else:
        args.replicas_to_average = list(range(args.num_subsets))

    best_paths = []
    for i in args.replicas_to_average:
        for j_id in args.resume_from + [args.job_id]:
            paths = glob.glob(
                os.path.join(MODEL_OUTPUT_PATH, args.dataset, f"{j_id}_*_{i}.ckpt")
            )
            if paths:
                best_paths.append(paths[0])
                break

    if len(best_paths) == len(args.replicas_to_average) and all(
        best_paths[i] for i in range(len(args.replicas_to_average))
    ):  # Best run from resuming and this job
        models = [
            MultiBiSageModule.load_from_checkpoint(best_paths[i])
            for i in range(len(args.replicas_to_average))
        ]
    else:
        raise FileNotFoundError(
            f"Not enough checkpoint file(s) found or provided for job_id={args.job_id} and/or resume_from={args.resume_from}"
        )

    L.seed_everything(args.seed, workers=True)

    ensembled_model, trainer, best_callback, logger = setup_ensemble_objects(
        args, datamodule, models, run_name
    )

    if args.skip_replicas:
        ensembled_model.save_hyperparameters()

    if trainer is not None:
        if args.ensemble_method == "avg_params":
            rank_zero_info("Run test set with non-fintuned averaged model...")
            trainer.test(model=ensembled_model, datamodule=datamodule)
        elif args.ensemble_method == "avg_params_finetune":
            if not args.run_only_test:
                datamodule.set_replica()
                rank_zero_info("Finetune averaged model...")
                trainer.fit(model=ensembled_model, datamodule=datamodule)
                ckpt_path = best_callback.best_model_path
            else:
                paths = glob.glob(
                    os.path.join(
                        MODEL_OUTPUT_PATH,
                        args.dataset,
                        f"{args.resume_from[0]}_*_ensembled.ckpt",
                    )
                )
                assert (
                    args.resume_from and paths
                ), "No checkpoint found for ensemble when expected"
                ckpt_path = paths[0]

            rank_zero_info("Run test set with fintuned averaged model...")
            trainer.test(
                model=ensembled_model, datamodule=datamodule, ckpt_path=ckpt_path
            )
        else:
            raise NotImplementedError()
    else:
        test_f1 = ensembled_model.inference_test(
            datamodule.test_dataloader()
        )
        logger.log_hyperparams(ensembled_model.hparams)
        if test_f1 is not None:
            logger.log_metrics({"test_f1": test_f1})


if __name__ == "__main__":
    main()

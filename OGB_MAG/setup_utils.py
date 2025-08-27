import json
import os
from datetime import timedelta

import lightning as L
import torch
from constants import MODEL_OUTPUT_PATH, WORKING_DIRECTORY
from data_module import DataModule
from ensemble_module import EnsembleModule
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from multibisage_module import MultiBiSageModule


def setup_replica_objects(
    args,
    datamodule: DataModule,
    replica_idx: int,
    run_name=None,
    metric: str = "val_f1",
    mode: str = "max",
):
    best_callback = ModelCheckpoint(
        dirpath=os.path.join(MODEL_OUTPUT_PATH, args.dataset),
        filename=f"{args.job_id}_{{{metric}:.4f}}_{replica_idx}",
        monitor=metric,
        mode=mode,
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
    latest_callback = ModelCheckpoint(
        dirpath=WORKING_DIRECTORY,
        filename=f"snapshot_{args.dataset}_{args.job_id}_{replica_idx}",
        monitor="epoch",
        mode="max",
        every_n_epochs=1,
        save_top_k=1,
        every_n_train_steps=0,
    )

    wandb_logger = None
    wandb_logger = WandbLogger(
        name=f"{run_name}_{replica_idx}" if run_name else None,
        log_model=False,
        group=f"jobid={args.job_id}, b={args.n_sample_multiplier}, s={args.num_subsets}",
        project="multibisage",
    )
    if run_name is None:
        run_name = wandb_logger.experiment.name

    model = MultiBiSageModule(
        name=args.dataset,
        num_classes=datamodule.num_classes,
        raw_feat_dim=datamodule.num_features,
        raw_neigh_feat_dims=datamodule.neighbor_feature_dims,
        embedding_dim=args.embedding_dim,
        hidden_layer_dim=2 * args.embedding_dim,
        num_transformer_heads=args.num_heads,
        num_transformer_layers=args.num_layers,
        dropout=args.dropout,
        is_multilabel=datamodule.is_multilabel,
        # use_lr_scheduler=args.sampler_type != "uniform",
        train_label_weights=datamodule.train_label_weights,  # if not args.no_bootstrap else None,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        strategy=DDPStrategy(
            find_unused_parameters=False, timeout=timedelta(seconds=18000)
        ),
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        max_epochs=args.epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=100),
            latest_callback,
            best_callback,
            LearningRateMonitor(),
        ],
        logger=wandb_logger,
        num_sanity_val_steps=0,
        sync_batchnorm=True,
        check_val_every_n_epoch=1,
        use_distributed_sampler=False,
    )
    if trainer.global_rank == 0 and wandb_logger is not None:
        wandb_logger.experiment.config.update(
            json.loads(
                json.dumps(vars(args)),
                parse_int=str,
                parse_float=str,
                parse_constant=str,
            )
        )

    return model, trainer, best_callback, run_name


def setup_ensemble_objects(
    args,
    datamodule,
    models,
    run_name,
    metric: str = "val_f1",
    mode: str = "max",
):
    best_callback = ModelCheckpoint(
        dirpath=os.path.join(MODEL_OUTPUT_PATH, args.dataset),
        filename=f"{args.job_id}_{{{metric}:.4f}}_ensembled",
        monitor=metric,
        mode=mode,
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
    latest_callback = ModelCheckpoint(
        dirpath=WORKING_DIRECTORY,
        filename=f"snapshot_{args.dataset}_{args.job_id}_ensembled",
        monitor="epoch",
        mode="max",
        every_n_epochs=1,
        save_top_k=1,
        every_n_train_steps=0,
    )

    model = EnsembleModule(
        name=args.dataset,
        num_classes=datamodule.num_classes,
        models=models,
        ensemble_method=args.ensemble_method,
        is_multilabel=datamodule.is_multilabel,
        train_label_weights=datamodule.train_label_weights,
    )

    wandb_logger = WandbLogger(
        name=f"{run_name}_ensemble={args.ensemble_method}" if run_name else None,
        log_model=False,
        group=f"jobid={args.job_id}, b={args.n_sample_multiplier}, s={args.num_subsets}",
        project="multibisage",
    )

    trainer = None
    if "avg_params" in args.ensemble_method:
        trainer = L.Trainer(
            accelerator="gpu",
            strategy=DDPStrategy(
                find_unused_parameters=False, timeout=timedelta(seconds=18000)
            ),
            devices=args.num_gpus,
            num_nodes=args.num_nodes,
            max_epochs=args.ensemble_epochs,
            callbacks=[
                TQDMProgressBar(refresh_rate=100),
                latest_callback,
                best_callback,
                LearningRateMonitor(),
            ],
            logger=wandb_logger,
            num_sanity_val_steps=0,
            sync_batchnorm=True,
            check_val_every_n_epoch=1,
            use_distributed_sampler=False,
        )
        if trainer.global_rank == 0:
            wandb_logger.experiment.config.update(
                json.loads(
                    json.dumps(vars(args)),
                    parse_int=str,
                    parse_float=str,
                    parse_constant=str,
                )
            )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    return model, trainer, best_callback, wandb_logger

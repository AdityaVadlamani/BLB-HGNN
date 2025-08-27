from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import Callable

import torch
from sampler import BLBSampler


class BLBWrapper:
    def __init__(self) -> None:
        pass

    def execute_with_blb(
        self, fn_main: Callable[[Namespace, BLBSampler], None], args: Namespace
    ):
        models = []
        for replica_idx in range(args.s):
            args.replica_idx = replica_idx
            models.append(
                fn_main(
                    args,
                )
            )

        with torch.no_grad():
            state_dicts = [m.state_dict() for m in models]

            averaged_state_dict = {}
            for key in state_dicts[0]:
                averaged_state_dict[key] = torch.sum(
                    torch.stack([s[key] * 1 / args.s for s in state_dicts]), axis=0
                )

            averaged_model = deepcopy(models[0])
            averaged_model.load_state_dict(averaged_state_dict)

        args.use_blb = False
        fn_main(args, averaged_model)
        args.use_blb = True

    def execute_without_blb(
        self, fn_main: Callable[[Namespace, BLBSampler], None], args: Namespace
    ):
        fn_main(
            args,
        )


def parse_wrapper_args(args=None):
    parser = ArgumentParser(description="BLB-GNN", allow_abbrev=False)

    parser.add_argument("--s", type=int, default=2, help="The number of subsets.")

    parser.add_argument("--b", type=float, default=1, help="Fraction of data.")

    parser.add_argument(
        "--b_rep", type=float, default=0.75, help="Frction of fraction of data."
    )

    parser.add_argument(
        "--use_dynamic_replica_set_frac",
        action="store_true",
        default=False,
        help="Set b_rep = 1 / s",
    )

    parser.add_argument(
        "--use_blb",
        action="store_true",
        default=False,
        help="Whether to use BLB",
    )

    return parser.parse_known_args(args)


if __name__ == "__main__":
    blb_args, extra_args = parse_wrapper_args()
    if blb_args.use_dynamic_replica_set_frac:
        blb_args.b_rep = 1 / blb_args.s
    wrapper = BLBWrapper()

    from SeHGNN.ogbn.main import main, parse_args

    args, _ = parse_args(extra_args)

    assert (
        len(set(vars(blb_args).keys()) & set(vars(args).keys())) == 0
    ), f"None of BLB keys: {list(blb_args.keys())} should be in program args"

    args = Namespace(**vars(blb_args), **vars(args))

    # Params suggested by respective code bases for training, but with no extra embeddings and single-stage training
    if args.dataset == "ogbn-mag":
        if blb_args.use_blb:
            args.stages = [50]
        else:
            args.stages = [100]

        args.seeds = [0, 1, 2]
        args.extra_embeddings = ""

    print(f"ARGS: {args}")

    for seed in args.seeds:
        args.seed = seed
        print("Restart with seed =", seed)
        if args.use_blb:
            wrapper.execute_with_blb(main, args)
        else:
            wrapper.execute_without_blb(main, args)

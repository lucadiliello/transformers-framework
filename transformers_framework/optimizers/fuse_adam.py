from typing import Generator

from transformers_framework.optimizers.optimizer import Optimizer
from transformers_framework.utilities.arguments import FlexibleArgumentParser
from transformers_framework.utilities.classes import ExtendedNamespace
from transformers_framework.utilities.strategy import check_strategy

from deepspeed.ops.adam import FusedAdam


class FuseAdamOptimizer(Optimizer, FusedAdam):

    def __init__(self, hyperparameters: ExtendedNamespace, named_parameters: Generator):
        r""" First hyperparameters argument to SuperOptimizer, other args for FusedAdam. """

        if not check_strategy(hyperparameters.strategy, "deepspeed"):
            raise ValueError("`FuseAdamOptimizer` should be used only with `deepspeed` strategy.")

        super().__init__(
            hyperparameters,
            named_parameters,
            lr=hyperparameters.learning_rate,
            bias_correction=True,
            eps=hyperparameters.fuse_adam_epsilon,
            betas=hyperparameters.fuse_adam_betas,
            amsgrad=hyperparameters.fuse_adam_amsgrad,
            set_grad_none=True,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--fuse_adam_epsilon', type=float, default=1e-8)
        parser.add_argument('--fuse_adam_betas', nargs=2, type=float, default=[0.9, 0.999])
        parser.add_argument('--fuse_adam_amsgrad', action='store_true')

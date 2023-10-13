from transformers_framework.utilities.classes import ExtendedNamespace
from typing import Generator

from torch.optim import AdamW

from transformers_framework.optimizers.optimizer import Optimizer
from transformers_framework.utilities.arguments import FlexibleArgumentParser


class AdamWOptimizer(Optimizer, AdamW):

    def __init__(self, hyperparameters: ExtendedNamespace, named_parameters: Generator):
        r""" First hyperparameters argument to SuperOptimizer, other args for AdamW. """
        super().__init__(
            hyperparameters,
            named_parameters,
            lr=hyperparameters.learning_rate,
            eps=hyperparameters.adamw_epsilon,
            betas=hyperparameters.adamw_betas,
            amsgrad=hyperparameters.adamw_amsgrad,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--adamw_epsilon', type=float, default=1e-8)
        parser.add_argument('--adamw_betas', nargs=2, type=float, default=[0.9, 0.999])
        parser.add_argument('--adamw_amsgrad', action='store_true')

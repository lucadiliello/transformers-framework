from argparse import Namespace
from typing import Generator

from transformers.optimization import Adafactor

from transformers_framework.optimizers.optimizer import Optimizer
from transformers_framework.utilities.arguments import FlexibleArgumentParser


class AdafactorOptimizer(Optimizer, Adafactor):

    def __init__(self, hyperparameters: Namespace, named_parameters: Generator):
        r""" First hyperparameters argument to SuperOptimizer, other args for Adafactor. """
        super().__init__(
            hyperparameters,
            named_parameters,
            lr=hyperparameters.learning_rate,
            eps=hyperparameters.adafactor_epsilon,
            clip_threshold=hyperparameters.adafactor_clip_threshold,
            beta1=hyperparameters.adafactor_beta1,
            decay_rate=hyperparameters.adafactor_decay_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--adafactor_epsilon', nargs=2, type=float, default=(1e-30, 1e-3))
        parser.add_argument('--adafactor_beta1', type=float, default=-0.8)
        parser.add_argument('--adafactor_decay_rate', type=float, default=-0.8)
        parser.add_argument('--adafactor_clip_threshold', type=float, default=1.0)

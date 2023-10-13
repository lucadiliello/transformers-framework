from transformers_framework.utilities.classes import ExtendedNamespace
from typing import Generator

from transformers_framework.utilities.arguments import FlexibleArgumentParser
from transformers_framework.utilities.optimizers import get_parameters_grouped_for_weight_decay


class Optimizer:
    r""" High level interface for optimizers to be used with transformers.
    Adds methods to define hyperparameters from the command line. """

    def __init__(self, hyperparameters: ExtendedNamespace, named_parameters: Generator, *args, **kwargs):
        r""" First argument should always be the hyperparameters namespace, other arguments are
        optimizer-specific. """
        grouped_parameters = get_parameters_grouped_for_weight_decay(
            named_parameters, weight_decay=hyperparameters.weight_decay
        )
        super().__init__(grouped_parameters, *args, **kwargs)  # forward init to constructor of real optimizer
        self.hyperparameters = hyperparameters

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        r""" Add here the hyperparameters used by your optimizer. """
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=0.0)

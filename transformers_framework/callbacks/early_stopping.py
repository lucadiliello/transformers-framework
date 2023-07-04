from argparse import Namespace

from lightning.pytorch.callbacks import EarlyStopping as _EarlyStopping

from transformers_framework.utilities.arguments import FlexibleArgumentParser


class EarlyStopping(_EarlyStopping):

    def __init__(self, hyperparameters: Namespace):
        if hyperparameters.monitor is None:
            raise ValueError("cannot use early_stopping without a monitored variable")

        super().__init__(
            monitor=hyperparameters.monitor,
            patience=hyperparameters.patience,
            verbose=True,
            mode=hyperparameters.monitor_direction,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        parser.add_argument(
            '--patience',
            type=int,
            default=5,
            required=False,
            help="Number of non-improving validations to wait before early stopping"
        )

from typing import Optional

from torchmetrics import Metric

from transformers_framework.utilities.arguments import FlexibleArgumentParser
from transformers_framework.utilities.logging import rank_zero_debug, rank_zero_info


class MetricsMixin:

    def setup(self, stage: Optional[str] = None):
        r""" Just check metrics are defined correctly. """
        rank_zero_info(f"Running setup for stage {stage}")

        for name, module in self.named_children():
            if isinstance(module, Metric):
                if not (
                    name.startswith('train')
                    or name.startswith('valid')
                    or name.startswith('test')
                ):
                    raise ValueError("All metrics in the model must start with 'train', 'valid' or 'test' ")

    def reset_metrics(self, stage: str):
        r""" Reset all metrics in model for the given stage. """
        rank_zero_debug(f"Resetting {stage} metrics")
        for name, module in self.named_children():
            if isinstance(module, Metric) and name.startswith(stage):
                module.reset()

    def on_train_epoch_start(self):
        r""" Reset training metrics. """
        self.reset_metrics('train')

    def on_validation_epoch_start(self):
        r""" Reset validation metrics. """
        self.reset_metrics('valid')

    def on_test_epoch_start(self):
        r""" Reset test metrics. """
        self.reset_metrics('test')

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        ...

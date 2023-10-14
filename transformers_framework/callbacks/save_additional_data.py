import os
import pickle
from typing import Any, Dict, List

import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback

from transformers_framework.utilities.arguments import FlexibleArgumentParser
from transformers_framework.utilities.classes import ExtendedNamespace
from transformers_framework.utilities.logging import rank_zero_only, rank_zero_warn


DEFAULT_DATA_DIR = "additional"


class SaveDataCallback(Callback):
    r"""
    This class allow to save additional data at each step or epoch end.

    Command line args:
    `--save_additional_data_interval`: Save additional data every given steps.
        A None value means save only at the end of each training epoch.
    """

    def __init__(self, hyperparameters: ExtendedNamespace, attribute_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyperparameters = hyperparameters
        self.destination = os.path.join(hyperparameters.output_dir, DEFAULT_DATA_DIR, hyperparameters.name)
        self.attribute_name = attribute_name

    def save(self, data: Any, filepath: str):
        r""" Save arbitrary object to disk. """
        if isinstance(data, torch.Tensor):
            torch.save(data.clone().detach().cpu(), filepath + ".pt")
        elif isinstance(data, np.ndarray):
            np.save(filepath + ".npy", data)
        else:
            with open(filepath + ".obj", "wb") as fo:
                pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)

    def save_data(self, pl_module: LightningModule, epoch: int = None, step: int = None):
        r""" Called when the a checkpoint should be saved. """
        basename = self.attribute_name
        if epoch is not None:
            basename += f"_epoch={epoch}"
        if step is not None:
            basename += f"_step={step}"

        filepath = os.path.join(self.destination, basename)

        # save data only if they are defined
        if hasattr(pl_module, self.attribute_name):
            self.save(
                data=getattr(pl_module, self.attribute_name),
                filepath=filepath,
            )

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        r""" Check model can be saved and save hyperparameters to understand what kind of experiment it was. """

        if not os.path.isdir(self.destination):
            os.makedirs(self.destination)

        if not hasattr(pl_module, self.attribute_name):
            rank_zero_warn(
                f"LightningModule {pl_module.__class__.__name__} has no "
                f"`{self.attribute_name}` attribute, then it will not be saved."
            )

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: List[Any], batch: Dict, batch_idx: int
    ):
        r"""Called when the training batch ends. """

        # save only on last accumulated batch
        if ((batch_idx + 1) % trainer.accumulate_grad_batches) != 0:
            return

        # save only when global step is multiple of save_additional_data_interval
        if (
            (self.hyperparameters.save_additional_data_interval is None)
            or (trainer.global_step % self.hyperparameters.save_additional_data_interval) != 0
        ):
            return

        self.save_data(pl_module, epoch=trainer.current_epoch, step=trainer.global_step)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        r"""Called when the train epoch ends."""
        self.save_data(pl_module, epoch=trainer.current_epoch, step=trainer.global_step)

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        r""" Add callback_specific arguments to parser. """
        parser.add_argument(
            '--save_additional_data_interval',
            type=int,
            required=False,
            default=None,
            help="Save pre_trained models every steps. A None value means save only at the end of each epoch."
        )

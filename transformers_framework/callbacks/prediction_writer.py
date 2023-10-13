import os
from typing import Any, List

import lightning.pytorch as pl
import torch
from datasets import Dataset
from lightning.pytorch.callbacks.prediction_writer import BasePredictionWriter
from lightning_fabric.utilities.rank_zero import rank_zero_only

from transformers_framework.utilities.classes import ExtendedNamespace
from transformers_framework.utilities.distributed import sync_data_distributed
from transformers_framework.utilities.logging import rank_zero_info, rank_zero_warn


class PredictionsWriter(BasePredictionWriter):

    def __init__(self, hyperparameters: ExtendedNamespace, destination: str):
        super().__init__(write_interval='epoch')
        self.destination = destination
        self.hyperparameters = hyperparameters

        if os.path.isdir(self.destination):
            rank_zero_warn(f"Destination folder '{self.destination}' for predictions does already exist!")

    @rank_zero_only
    def save_predictions(self, data: List[Any]):
        res = Dataset.from_dict(dict(scores=data))
        rank_zero_info("Saving predictions to disk...")
        res.save_to_disk(self.destination)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        r""" Collect predictions from all process, sort them and write them to disk. """
        batch_indices = [torch.tensor(batch_idx) for batch_idx in batch_indices]

        # collect data from all processes
        predictions = sync_data_distributed(predictions, concat=True)
        batch_indices = sync_data_distributed(batch_indices, concat=True)

        # concate on tensor of single dimension
        predictions = torch.cat(predictions, dim=0)
        batch_indices = torch.cat(batch_indices, dim=0)

        # sort predictions based on original example index
        predictions = predictions[batch_indices.argsort(dim=0, descending=False)]

        # save to disk only on rank 0
        self.save_predictions(predictions.cpu().detach().tolist())

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.interval.on_epoch:
            return
        epoch_batch_indices = trainer.predict_loop.epoch_batch_indices[0]
        self.write_on_epoch_end(trainer, pl_module, trainer.predict_loop.predictions, epoch_batch_indices)

import os
import shutil
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

from transformers_framework.datasets.iterable_dataset import IterableDataset
from transformers_framework.datasets.map_dataset import MapDataset
from transformers_framework.pipelines.pipeline.pipeline import Pipeline
from transformers_framework.utilities.arguments import FlexibleArgumentParser
from transformers_framework.utilities.classes import ExtendedNamespace
from transformers_framework.utilities.datamodules import TrainerFn_to_Names
from transformers_framework.utilities.datasets import load_dataset_from_anywhere
from transformers_framework.utilities.hash import hash_function, hash_string
from transformers_framework.utilities.logging import rank_zero_info, rank_zero_warn
from transformers_framework.utilities.methods import is_overidden


class ArrowDataModule(LightningDataModule):
    r"""
    ArrowDataModule is backed by arrow dataset from huggingface `datasets`.
    It implements some simple methods to check whether training, validation, testing or predictions is required.
    Moreover, it adds to the command line parameters the basic arguments used by Dataset,
    such as `batch_size` and `num_workers`.

    Example:

    >>> if datamodule.do_train():
    >>>     trainer.fit(model, datamodule=datamodule)

    >>> if datamodule.do_test():
    >>>     trainer.test(model, datamodule=datamodule)
    """

    # ######################## should perform stage ####################

    def do_train(self):
        return self.hyperparameters[f'{TrainerFn_to_Names[TrainerFn.FITTING]}_dataset'] is not None

    def do_validation(self):
        return self.hyperparameters[f'{TrainerFn_to_Names[TrainerFn.VALIDATING]}_dataset'] is not None

    def do_test(self):
        return self.hyperparameters[f'{TrainerFn_to_Names[TrainerFn.TESTING]}_dataset'] is not None

    def do_predict(self):
        return self.hyperparameters[f'{TrainerFn_to_Names[TrainerFn.PREDICTING]}_dataset'] is not None

    def do_stage(self, stage: TrainerFn):
        return self.hyperparameters[f'{TrainerFn_to_Names[stage]}_dataset'] is not None

    # ######################## should reload data for stage ####################

    def do_reload_train_every_epoch(self):
        return self.hyperparameters[f'reload_{TrainerFn_to_Names[TrainerFn.FITTING]}_dataset_every_epoch']

    def do_reload_val_every_epoch(self):
        return self.hyperparameters[f'reload_{TrainerFn_to_Names[TrainerFn.VALIDATING]}_dataset_every_epoch']

    def do_reload_test_every_epoch(self):
        return self.hyperparameters[f'reload_{TrainerFn_to_Names[TrainerFn.TESTING]}_dataset_every_epoch']

    def do_reload_predict_every_epoch(self):
        return self.hyperparameters[f'reload_{TrainerFn_to_Names[TrainerFn.PREDICTING]}_dataset_every_epoch']

    def do_reload_stage_every_epoch(self, stage: TrainerFn):
        return self.hyperparameters[f'reload_{TrainerFn_to_Names[stage]}_dataset_every_epoch']

    # ######################## main methods ####################

    def __init__(self, hyperparameters: ExtendedNamespace, trainer: Trainer, model: Pipeline):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.trainer = trainer
        self.model = model

        # automatically set num workers as the number of CPUs divided by the number of GPUs
        if self.hyperparameters.num_workers is None:
            self.hyperparameters.num_workers = cpu_count() // self.hyperparameters.devices
            rank_zero_warn(f"Automatically set `num_workers` equal to {self.hyperparameters.num_workers}")

        # prepare data on local_rank 0 of each node or only on global zero (distributed file system)
        if not self.hyperparameters.prepare_data_per_node and self.trainer.num_nodes > 1:
            rank_zero_warn(
                "Training using more than a single machine but preparing data only on global rank 0."
                "Make sure to have a distributed file system."
            )

        if (
            any(
                self.hyperparameters[f'reload_{stage_name}_dataset_every_epoch']
                for stage_name in TrainerFn_to_Names.values()
            )
            and self.hyperparameters.reload_dataloaders_every_n_epochs != 1
        ):
            raise ValueError(
                "Set `reload_dataloaders_every_n_epochs=1` when using any of reload_{stage_name}_dataset_every_epoch."
            )

        self.prepare_data_per_node = self.hyperparameters.prepare_data_per_node
        self.prepare_data_user_defined = is_overidden(self.model.preprocess)

        # clean before experiment
        if self.rank_should_prepare_data():
            main_path = self.get_temporary_directory()
            rank_zero_warn(f"Cleaning up temporary directory {main_path}")
            shutil.rmtree(main_path, ignore_errors=True)

    def rank_should_prepare_data(self):
        r""" Guard to run code only on local/global (based on prepare_data_per_node) rank in distributed setting. """
        if dist.is_available() and dist.is_initialized():
            if self.prepare_data_per_node:
                return self.trainer.local_rank == 0
            else:
                return self.trainer.global_rank == 0
        return True

    def get_temporary_directory(self) -> str:
        r""" Get main temporary directory for all datasets. """
        main_path = os.path.join(
            self.hyperparameters.temporary_data_folder, self.hyperparameters.name,
            hash_function(self.model.preprocess)[:24] + "_" + hash_string(self.model.__class__.__name__)[:24]
        )
        os.makedirs(main_path, exist_ok=True)
        return main_path

    def get_temporary_dataset_directory(self, stage: TrainerFn) -> str:
        r""" Create and get path of temporary directory for preprocessing datasets. """
        main_path = self.get_temporary_directory()
        dataset_path = TrainerFn_to_Names[stage]
        if self.do_reload_stage_every_epoch(stage):
            dataset_path += f"_epoch{self.trainer.current_epoch}"
        return os.path.join(main_path, dataset_path)

    def preprocessing(self, stage: TrainerFn):
        r""" Preprocess datasets. This should be run only on the local_rank=0.
        This will not set any state. It will just save the dataset to the pre-defined location,
        eventually adding a suffix "_epoch0" is dataset reloading is set for that stage.
        """
        # user defined a custom preprocessing function
        if self.rank_should_prepare_data() and self.prepare_data_user_defined:
            rank_zero_info("User defined a custom preprocess function inside pipeline...")

            if self.do_stage(stage):
                save_path = self.get_temporary_dataset_directory(stage)
                stage_name = TrainerFn_to_Names[stage]

                if not os.path.isdir(save_path):
                    rank_zero_info(f"Preprocessing {stage_name} dataset")
                    dataset, _ = load_dataset_from_anywhere(
                        self.hyperparameters[f'{stage_name}_dataset'],
                        config=self.hyperparameters[f'{stage_name}_config'],
                        keep_in_memory=self.hyperparameters.keep_in_memory,
                    )

                    dataset = self.model.preprocess(
                        dataset,
                        num_workers=self.hyperparameters.prepare_data_workers,
                        batch_size=self.hyperparameters.prepare_data_batch_size,
                        load_from_cache_file=not self.hyperparameters.prepare_data_do_not_use_cache,
                    )
                    rank_zero_info(f"Saving preprocessed dataset to {save_path}")
                    dataset.save_to_disk(save_path)
                else:
                    rank_zero_info(f"Reusing already preprocessed dataset for stage {stage_name}")

        # wait for all processes to have reached this point to avoid early loading of datasets not yet created
        self.trainer.strategy.barrier()

    def load_dataset(self, stage: TrainerFn):
        r""" Load a dataset given the stage name. """
        rank_zero_info(f"Loading {stage.value} dataset...")

        # compute cached dataset
        dataset_path = (
            self.get_temporary_dataset_directory(stage)
            if self.prepare_data_user_defined
            else self.hyperparameters[f"{TrainerFn_to_Names[stage]}_dataset"]
        )
        config = (
            None if self.prepare_data_user_defined else self.hyperparameters[f"{TrainerFn_to_Names[stage]}_config"]
        )

        # load from anywhere and log info
        dataset, location = load_dataset_from_anywhere(
            dataset_path,
            config=config,
            keep_in_memory=self.hyperparameters.keep_in_memory,
            shard=self.hyperparameters[f'{TrainerFn_to_Names[stage]}_shard_dataset'],
        )
        rank_zero_info(f"Loaded {stage.value.capitalize()} dataset from {location}. Length: {len(dataset)}")

        # create pytorch dataset instance
        dataset = (IterableDataset if self.hyperparameters.iterable else MapDataset)(dataset)
        return dataset

    def default_dataloader(self, dataset: Dataset, shuffle: bool = False, batch_size: int = None):
        r""" Return a dataloader with all usual default parameters. """

        if self.hyperparameters.iterable and shuffle:
            raise ValueError("Found shuffle=True while using IterableDataset")

        if batch_size is None:
            batch_size = self.hyperparameters.batch_size

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.hyperparameters.num_workers,
            pin_memory=not self.hyperparameters.disable_memory_pinning,
            collate_fn=self.model.collate_fn,
            shuffle=shuffle,
            multiprocessing_context=(
                'fork' if torch.backends.mps.is_available() and self.hyperparameters.num_workers > 0 else None
            ),
        )

    def train_dataloader(self):
        r""" Return the training dataloader. """
        if not self.do_train():
            return None
        self.preprocessing(TrainerFn.FITTING)
        self.train_dataset = self.load_dataset(TrainerFn.FITTING)
        return self.default_dataloader(self.train_dataset, shuffle=not self.hyperparameters.iterable)

    def val_dataloader(self):
        r""" Return the validation dataloader. """
        if not self.do_validation():
            return None
        self.preprocessing(TrainerFn.VALIDATING)
        self.valid_dataset = self.load_dataset(TrainerFn.VALIDATING)
        return self.default_dataloader(
            self.valid_dataset, shuffle=False, batch_size=self.hyperparameters.eval_batch_size
        )

    def test_dataloader(self):
        r""" Return the test dataloader. """
        if not self.do_test():
            return None
        self.preprocessing(TrainerFn.TESTING)
        self.test_dataset = self.load_dataset(TrainerFn.TESTING)
        return self.default_dataloader(
            self.test_dataset, shuffle=False, batch_size=self.hyperparameters.eval_batch_size
        )

    def predict_dataloader(self):
        r""" Return the predict dataloader. """
        if not self.do_predict():
            return None
        self.preprocessing(TrainerFn.PREDICTING)
        self.predict_dataset = self.load_dataset(TrainerFn.PREDICTING)
        return self.default_dataloader(
            self.predict_dataset, shuffle=False, batch_size=self.hyperparameters.eval_batch_size
        )

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        r""" Transfer batch to device. """
        if isinstance(device, str):
            device = torch.device(device)

        return {
            key: torch.from_numpy(value).to(device=device).contiguous() if isinstance(value, np.ndarray) else value
            for key, value in batch.items()
        }

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        # dataloaders stuff
        parser.add_argument('--num_workers', type=int, required=False, default=None, help='Number of workers')
        parser.add_argument(
            '--disable_memory_pinning', action="store_true", help="Disable memory pinning in dataloader"
        )
        parser.add_argument('--batch_size', type=int, required=True, help="Train/valid/test batch size")
        parser.add_argument(
            '--eval_batch_size', type=int, required=False, default=None, help="valid/test batch size if different"
        )

        parser.add_argument('--iterable', action="store_true")

        # preprocessing stuff
        parser.add_argument(
            '--prepare_data_per_node',
            action="store_true",
            help="Preprocess data only on rank 0 node 0 (shared FS).",
        )
        parser.add_argument(
            '--prepare_data_workers',
            type=int,
            required=False,
            default=cpu_count(),
            help="Number of CPUs for dataset preprocessing.",
        )
        parser.add_argument(
            '--prepare_data_batch_size',
            type=int,
            required=False,
            default=1000,
            help="Batch size for dataset preprocessing.",
        )
        parser.add_argument('--prepare_data_do_not_use_cache', action="store_true")

        # datasets stuff
        parser.add_argument(
            '--temporary_data_folder',
            type=str,
            required=False,
            default="/science/lucadiliello/.cache/transformers_framework/datasets",
        )
        parser.add_argument(
            '--keep_in_memory', action="store_true", help="Keep all arrow datasets in memory"
        )
        for stage_name in TrainerFn_to_Names.values():
            parser.add_argument(
                f'--{stage_name}_dataset',
                type=str,
                required=False,
                default=None,
                help=f"Path to {stage_name} dataset dump",
            )
            parser.add_argument(
                f'--{stage_name}_config',
                type=str,
                required=False,
                default=None,
                help=f"Path to {stage_name} dataset config",
            )
            parser.add_argument(
                f'--{stage_name}_shard_dataset',
                type=int,
                required=False,
                default=None,
                help=f"Reduce {stage_name} dataset that number of times",
            )
            parser.add_argument(
                f'--reload_{stage_name}_dataset_every_epoch',
                action="store_true",
                help=f"Path to {stage_name} dataset config",
            )

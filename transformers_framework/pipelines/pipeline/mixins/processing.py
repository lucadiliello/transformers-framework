from typing import Any, Dict, List, Tuple

from datasets import Dataset

from transformers_framework.utilities.arguments import FlexibleArgumentParser
from transformers_framework.utilities.collate import collate_flexible_numpy_fn
from transformers_framework.utilities.functional import shrink_batch
from transformers_framework.utilities.methods import native


class ProcessingMixin:

    MODEL_INPUT_NAMES_TO_REDUCE: List[Tuple[str]] = None

    @native
    def preprocess(
        self,
        dataset: Dataset,
        num_workers: int = None,
        batch_size: int = None,
        load_from_cache_file: bool = True,
    ) -> Dataset:
        r""" Preprocess a dataset. Useful for tasks like MR, NER and POS. """
        ...

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Postprocess a single sample. Samples will be extracted from the dataset returned above.
        This operation will be performed in dataloader's workers.
        Please use numpy as much as you can to speed up operations.
        """
        return sample

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        r""" Merge together samples drawn from dataset after `postprocess`. This method should return
        numpy arrays of torch tensors. This is a good place for padding and shrinking tensors. """
        batch = [self.postprocess(sample) for sample in batch]
        batch = collate_flexible_numpy_fn(batch)

        if self.MODEL_INPUT_NAMES_TO_REDUCE:
            for group in self.MODEL_INPUT_NAMES_TO_REDUCE:
                if group[0] not in batch or batch[group[0]] is None:
                    raise ValueError(
                        f"requested to reduce {group} but batch does not contain"
                        f"at least the first key {group[0]} or is None. available keys in batch: {tuple(batch.keys())}"
                    )
                self.shrink_batch(batch, group)
        return batch

    def shrink_batch(self, batch: Dict[str, Any], keys: List[str] = None, pad_token_id: int = None):
        r""" Remove data on the sequence length dimension in the positions where every example is padded. """
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id

        if self.hyperparameters.compile:
            return

        # remove from group keys that are not available in the batch or that are None
        keys = [keys[0]] + list(set(keys[1:]).intersection(batch.keys()))
        keys = [k for k in keys if batch[k] is not None]

        shrink_batch(
            batch=batch,
            keys=keys,
            pad_token_id=pad_token_id,
            shrink_to_multiples_of=8 if self.hyperparameters.precision in (16, '16', 'bf16') else None,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        # data preparation
        parser.add_argument(
            '--max_sequence_length', type=int, nargs="+", required=True, help="Input max sequence length(s)"
        )

from typing import Optional

import torch
from transformers.generation.stopping_criteria import StoppingCriteria


class StoppingCriteriaList(list):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(criteria(input_ids, scores, **kwargs) for criteria in self)

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
        return None


class MaxLengthCriteria(StoppingCriteria):
    r"""
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
    """

    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, last_logits_positions: torch.LongTensor
    ) -> bool:
        return last_logits_positions.min() + 1 >= self.max_length

from typing import Dict, List

import torch
from lightning.pytorch.utilities.memory import garbage_collection_cuda
from torch import nn

from transformers_framework.utilities import IGNORE_IDX


def shuffle_tensor(_tensor: torch.Tensor, dim: int = -1):
    r""" Randomly shuffle tensor along dim. """
    indices = torch.randperm(_tensor.shape[dim])
    _tensor = _tensor.index_select(dim=dim, index=indices)
    return _tensor


def torch_unsorted_segment_sum(
    segment_ids: torch.Tensor, data: torch.Tensor = None, length: int = None
) -> torch.Tensor:
    r""" Equivalent of `tf.unsorted_segment_sum`. """
    if data is None:
        data = torch.ones_like(segment_ids)

    if length is None:
        length = segment_ids.max() + 1

    assert segment_ids.shape == data.shape  # nosec

    res = torch.zeros(length, dtype=data.dtype)
    res = res.scatter_add(0, segment_ids, data)
    return res


def combine_losses(losses: List[torch.Tensor], weigths: List[float] = None):
    r""" Combine multiple losses weighted by weights. """

    assert any(loss is not None for loss in losses), "expected at least a non None loss"  # nosec

    if weigths is None:
        weigths = [1.0] * len(losses)

    return sum(loss * weight for loss, weight in zip(losses, weigths) if loss is not None)


def pad_and_transpose_last_two_dims(hidden_states_padded: torch.Tensor, padding: int):
    r""" Pads rows and then flips rows and columns. """
    hidden_states_padded = nn.functional.pad(
        hidden_states_padded, padding
    )  # padding value is not important because it will be overwritten
    hidden_states_padded = hidden_states_padded.view(
        *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
    )
    return hidden_states_padded


def pad_and_diagonalize(chunked_hidden_states: torch.Tensor):
    r"""
    Shift every row 1 step right, converting columns into diagonals.

    Example::

            chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                    -1.8348,  0.7672,  0.2986,  0.0285,
                                    -0.7584,  0.4206, -0.0405,  0.1599,
                                    2.0514, -1.1600,  0.5372,  0.2629 ]
            window_overlap = num_rows = 4
            (pad & diagonalize) =>
            [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
            0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
            0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
            0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
    """
    total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
    chunked_hidden_states = nn.functional.pad(
        chunked_hidden_states, (0, window_overlap + 1)
    )
    # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
    # Padding value is not important because it'll be overwritten

    chunked_hidden_states = chunked_hidden_states.view(
        total_num_heads, num_chunks, -1
    )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
    chunked_hidden_states = chunked_hidden_states[
        :, :, :-window_overlap
    ]  # total_num_heads x num_chunks x window_overlap*window_overlap
    chunked_hidden_states = chunked_hidden_states.view(
        total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
    )
    chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    return chunked_hidden_states


def chunk(hidden_states: torch.Tensor, window_overlap: int):
    r""" Convert into overlapping chunks. Chunk size = 2w, overlap size = w. """

    # non-overlapping chunks of size = 2w
    hidden_states = hidden_states.view(
        hidden_states.size(0),
        hidden_states.size(1) // (window_overlap * 2),
        window_overlap * 2,
        hidden_states.size(2),
    )

    # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
    chunk_size = list(hidden_states.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(hidden_states.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)


def mask_invalid_locations(input_tensor: torch.Tensor, affected_seq_len: int) -> torch.Tensor:
    beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    beginning_mask = beginning_mask_2d[None, :, None, :]
    ending_mask = beginning_mask.flip(dims=(1, 3))
    beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8
    ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    ending_mask = ending_mask.expand(ending_input.size())
    ending_input.masked_fill_(ending_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8


def single_token_to_multi_token_labels(labels: torch.Tensor, multi_token_number: int) -> torch.Tensor:
    r""" Convert labels from single dim to bi-dim. """

    # create output array
    res = torch.full(
        size=(*labels.shape, multi_token_number), dtype=torch.int64, fill_value=IGNORE_IDX, device=labels.device
    )

    # padding size
    pad_size = multi_token_number - labels.size(1) % multi_token_number
    if pad_size == multi_token_number:
        pad_size = 0

    # pad tensor
    labels = torch.nn.functional.pad(labels, (0, pad_size), mode='constant', value=IGNORE_IDX)
    labels = labels.view(labels.shape[0], -1, multi_token_number)

    res[:, ::multi_token_number] = labels

    return res


def split_batch(batch: Dict[str, torch.Tensor], size: int) -> List[Dict[str, torch.Tensor]]:
    r""" Split tensors in batch along first axis. """

    keys = list(batch.keys())

    return [
        {k: v for k, v in zip(keys, data)} for data in zip(*[torch.split(batch[k], size, dim=0) for k in keys])
    ]


def clean_device_cache():
    r""" Clean devices cache on different accelerators. """
    if torch.cuda.is_available():  # NVIDIA CUDA and AMD ROCm
        garbage_collection_cuda()
    elif torch.backends.mps.is_available():  # Apple MPS
        torch.mps.empty_cache()


def logits_to_binary_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    r""" Receives a tensor of floats with shape (batch_size,) and return a
    tensor with the same shape and integers in [0, 1].
    Applies a sigmoid and a comparison with the threshold.
    """
    preds = torch.sigmoid(logits)
    preds = (preds > threshold).to(dtype=torch.int64)
    return preds


def similarities_to_binary_prediction(similarities: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    r""" Receives a tensor of floats with shape (batch_size,) and return a
    tensor with the same shape and integers in [0, 1].
    Applies a comparison with the threshold.
    """
    preds = (similarities > threshold).to(dtype=torch.int64)
    return preds

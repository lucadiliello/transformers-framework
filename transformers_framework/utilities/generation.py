from typing import Dict, Iterable, List, Tuple

import torch


def remove_padding_end(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    r""" Remove all padding from `input_ids` on last dim.  """
    positions = (input_ids != pad_token_id).any(dim=0)
    input_ids = input_ids[..., positions]
    return input_ids


def get_last_logits_positions(input_ids: torch.Tensor, pad_token_id: int, return_mask: bool = False) -> torch.Tensor:
    r""" Get the positions of the next valid logits. This is necessary because with multi-token generation,
    each generated sequence so far can have a different length and doing logits[:, -1] is thus wrong.

    Parameters:
        input_ids (`torch.Tensor` of shape (batch_size, sequence_length)):
            The input_ids generated so far for which the last position will be searched.
        pad_token_id (int):
            The token that is used by the model for the generation

    Return:
        `a torch.Tensor` containing the positions of the first pad_token

    Example:
    >>> from torch import tensor
    >>> pad_token_id = 0
    >>> input_ids = tensor([
    ...     [0, 23, 44, 65, 11,  0,  0,  0],
    ...     [0, 11, 32, 66, 88, 11,  0,  0],
    ... ])

    >>> get_last_logits_positions(input_ids, pad_token_id, return_mask=False)
    ... tensor([
    ...     4,
    ...     5,
    ... ])

    >>> get_last_logits_positions(input_ids, pad_token_id, return_mask=True)
    ... tensor([
    ...     [False, False, False, False,  True, False, False, False],
    ...     [False, False, False, False, False,  True, False, False],
    ... ])
    """

    assert input_ids.dim() == 2, "`input_ids` must be of shape (batch_size, sequence_length)"

    # compute positions of last non pad token
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    positions = positions * (input_ids != pad_token_id)

    if return_mask:
        return positions == positions.max(dim=-1, keepdim=True).values
    else:
        return positions.max(dim=-1).values


def append_new_input_ids(input_ids: torch.Tensor, new_input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    r""" Aggregate a new token to existing input_ids. This should take care of the new padding setting after
    concatenation.

    Parameters:
        input_ids (`torch.Tensor` of shape (batch_size, sequence_length)):
            The input_ids generated so far for which the last position will be searched
        new_input_ids (`torch.Tensor` of shape (batch_size,)):
            The new generated token
        pad_token_id (int):
            The token that is used by the model for the generation

    Return:
        `a torch.Tensor` containing all the ids

    Example:
    >>> from torch import tensor
    >>> pad_token_id = 0
    >>> input_ids = tensor([
    ...     [ 0, 23, 44, 65, 11,  0,  0,  0],
    ...     [ 0, 11, 32, 66, 88, 11,  0,  0],
    ... ])
    >>> new_input_ids = tensor([
    ...     21,
    ...     15,
    ... ])

    >>> append_new_input_ids(input_ids, new_input_ids, pad_token_id)
    ... tensor([
    ...     [ 0, 23, 44, 65, 11, 21,  0,  0],
    ...     [ 0, 11, 32, 66, 88, 11, 15,  0],
    ... ])
    """

    assert input_ids.dim() == 2, "`input_ids` must be of shape (batch_size, sequence_length)"
    assert new_input_ids.dim() == 1, "`new_input_ids` must be of shape (batch_size, )"
    assert input_ids.shape[0] == new_input_ids.shape[0], "`input_ids` and `new_input_ids` must have the same batch size"

    # adding original input_ids
    positions = get_last_logits_positions(input_ids, pad_token_id, return_mask=False) + 1  # [ 5, 6 ]

    # check if we need to pad on the right or there is already space (padded positions) in the input_ids for new tokens
    if positions.max() >= input_ids.shape[1]:
        # increment sequence length by 1 adding
        input_ids = torch.cat([
            input_ids,
            new_input_ids.new_full(size=(new_input_ids.shape[0], 1), fill_value=pad_token_id),
        ], dim=1)

    input_ids = input_ids.scatter(1, positions.unsqueeze(-1), new_input_ids.unsqueeze(1))
    return input_ids


def extend_new_input_ids(input_ids: torch.Tensor, new_input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    r""" Aggregate new tokens to existing input_ids. This should take care of the new padding setting after
    concatenation.

    Parameters:
        input_ids (`torch.Tensor` of shape (batch_size, sequence_length)):
            The input_ids generated so far for which the last position will be searched
        new_input_ids (`torch.Tensor` of shape (batch_size, num_of_new_tokens)):
            The new generated tokens
        pad_token_id (int):
            The token that is used by the model for the generation

    Return:
        `a torch.Tensor` containing all the ids

    Example:
    >>> from torch import tensor
    >>> pad_token_id = 0
    >>> input_ids = tensor([
    ...     [ 0, 23, 44, 65, 11,  0,  0,  0],
    ...     [ 0, 11, 32, 66, 88, 11,  0,  0],
    ... ])
    >>> new_input_ids = tensor([
    ...     [21, 12,  0,  0],
    ...     [15, 17,  3,  0],
    ... ])

    >>> aggregate_new_input_ids(input_ids, new_input_ids, pad_token_id)
    ... tensor([
    ...     [ 0, 23, 44, 65, 11, 21, 12,  0,  0,  0,  0,  0],
    ...     [ 0, 11, 32, 66, 88, 11, 15, 17,  0,  0,  0,  0],
    ... ])
    """

    assert input_ids.dim() == 2, "`input_ids` must be of shape (batch_size, sequence_length)"
    assert new_input_ids.dim() == 2, "`new_input_ids` must be of shape (batch_size, sequence_length)"
    assert input_ids.shape[0] == new_input_ids.shape[0], "`input_ids` and `new_input_ids` must have the same batch size"

    # adding original input_ids
    positions = get_last_logits_positions(input_ids, pad_token_id, return_mask=False) + 1  # [ 5, 6 ]

    # output tensor that we will eventually shrink
    res = torch.cat([input_ids, new_input_ids.new_full(size=new_input_ids.size(), fill_value=pad_token_id)], dim=1)

    index = torch.arange(new_input_ids.shape[1]).expand_as(new_input_ids).to(new_input_ids) + positions.unsqueeze(-1)
    # [[5, 6, 7, ...], [6, 7, 8, ...]]

    res.scatter_(1, index, new_input_ids)

    return res


def prepare_decoder_prediction_mask_for_generation(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    r""" Prepare prediction mask, which will be just a series of ones. """

    prediction_mask = torch.zeros_like(input_ids)
    prediction_mask[get_last_logits_positions(input_ids, pad_token_id=pad_token_id, return_mask=True)] = 1
    return prediction_mask


def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor) -> Dict[Tuple, List]:
    generated_ngrams = {}

    gen_tokens = prev_input_ids.tolist()
    for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
        prev_ngram_tuple = tuple(ngram[:-1])
        generated_ngrams[prev_ngram_tuple] = generated_ngrams.get(prev_ngram_tuple, []) + [ngram[-1]]

    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(ngram_size: int, prev_input_ids: torch.Tensor, cur_len: int) -> Iterable[int]:
    r""" Copied from fairseq for no_repeat_ngram in beam_search. """
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return []

    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids)
    banned_tokens = _get_generated_ngrams(generated_ngrams, prev_input_ids, ngram_size, cur_len)
    return banned_tokens

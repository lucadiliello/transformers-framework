from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.functional import pad_array, shift_tokens_right
from transformers_framework.utilities.numpy import (
    compress_spans_to_unique_tokens,
    correct_mask_to_cover_words,
    get_generator,
    random_spans_mask,
)
from transformers_framework.utilities.processors import (
    extend_mask_on_tails,
    generate_new_attention_mask,
    whole_word_tails_mask,
)


def masked_language_modeling(
    input_ids: npt.NDArray[np.int64],
    word_ids: npt.NDArray[np.int64],
    mask_token_id: int,
    vocab_size: int,
    probability: float = 0.15,
    specific_probabilities: List[float] = [0.8, 0.1, 0.1],
    whole_word_masking: bool = False,
    disable: bool = False,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r""" Select a random subsample of token and then either replace them with MASK,
    change them into random others of just leave them unchanged.

    Args:
        input_ids (`npt.NDArray[np.int64]`): list of original input ids
        word_ids (`npt.NDArray[np.int64]`): original word ids. None is used for special tokens
        mask_token_id (`int`): mask token id
        vocab_size (`int`): vocabulary size, used to sample replacements
        probability (`float`): fraction of tokens to mask/replace/leave unchanged
        specific_probabilities (`List[float]`): specific probabilities for mask/replace/leave unchanged
        whole_word_masking (`bool`): whether to do whole word masking

    Returns:
        A tuple with the masked input ids and the corresponding labels
    """

    if disable:
        return input_ids, None

    assert sum(specific_probabilities) == 1.0  # nosec

    # work with numpy array for faster masking
    masked_input_ids = np.copy(input_ids)
    masked_lm_labels = np.copy(input_ids)

    # We sample a few tokens in each sequence for masked-LM training
    # (with probability probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = np.full(masked_input_ids.shape, fill_value=probability, dtype=np.float32)

    # create whole work masking mask -> True if the token starts with ## (following token in composed words)
    if whole_word_masking:
        word_tails = whole_word_tails_mask(word_ids)
        # with whole word masking probability matrix should average probability over the entire word
        probability_matrix[word_tails == 1] = 0.0

    # set probability of special tokens to 0
    probability_matrix[word_ids == IGNORE_IDX] = 0.0

    # draw mask indices from bernoulli distribution
    masked_indices = get_generator().binomial(n=1, p=probability_matrix).astype(dtype=np.bool_)

    # with whole word masking, assure all tokens in a word are either all masked or not
    if whole_word_masking:
        masked_indices = extend_mask_on_tails(masked_indices, word_tails)

    masked_lm_labels[~masked_indices] = IGNORE_IDX    # We only compute loss on masked tokens

    # create array from which we will sample at once whether tokens are masked, replaced or left unchanged
    positions = get_generator().choice(3, size=len(masked_input_ids), p=specific_probabilities)

    indexes_to_mask = (positions == 0) & masked_indices
    indexes_to_replace = (positions == 1) & masked_indices
    # indexes_to_leave_unchanged = (positions == 2) & masked_indices

    # mask tokens
    masked_input_ids[indexes_to_mask] = mask_token_id

    # replace tokens
    new_tokens = get_generator().choice(vocab_size, size=len(masked_input_ids))
    masked_input_ids[indexes_to_replace] = new_tokens[indexes_to_replace]

    return masked_input_ids, masked_lm_labels


def random_token_substitution(
    input_ids: npt.NDArray[np.int64],
    word_ids: npt.NDArray[np.int64],
    vocab_size: int,
    probability: float = 0.15,
    whole_word_detection: bool = False,
    disable: bool = False,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r""" Select a random subsample of token and then replace them with other random tokens.

    Args:
        input_ids (`npt.NDArray[np.int64]`): list of original input ids
        word_ids (`npt.NDArray[np.int64]`): original word ids. None is used for special tokens
        vocab_size (`int`): vocabulary size, used to sample replacements
        probability (`float`): fraction of tokens to replace
        whole_word_detection (`bool`): whether to do whole word substitution

    Returns:
        A tuple with the replaced input ids and the corresponding labels
    """

    if disable:
        return input_ids, None

    replaced_input_ids = input_ids.copy()
    replaced_labels = np.full(input_ids.shape, fill_value=0, dtype=int)

    # We sample a few tokens in each sequence for TD training
    # (with probability probability defaults to 0.15 in ELECTRA)
    probability_matrix = np.full(replaced_input_ids.shape, fill_value=probability, dtype=np.float32)

    # create whole work masking mask -> True if the token starts with ## (following token in composed words)
    if whole_word_detection:
        word_tails = whole_word_tails_mask(word_ids)
        # with whole word masking probability matrix should average probability over the entire word
        probability_matrix[word_tails == 1] = 0.0

    special_tokens_mask = (word_ids == IGNORE_IDX)

    probability_matrix[special_tokens_mask] = 0.0
    replaced_labels[special_tokens_mask] = 0

    substituted_indices = get_generator().binomial(n=1, p=probability_matrix).astype(dtype=np.bool_)

    # with whole word masking, assure all tokens in a word are either all masked or not
    if whole_word_detection:
        substituted_indices = extend_mask_on_tails(substituted_indices, word_tails)

    # replace tokens
    new_tokens = get_generator().choice(vocab_size, size=len(replaced_input_ids))
    replaced_input_ids[substituted_indices] = new_tokens[substituted_indices]
    replaced_labels[substituted_indices] = 1

    return replaced_input_ids, replaced_labels


def denoising_bart_version(
    input_ids: npt.NDArray[np.int64],
    word_ids: npt.NDArray[np.int64],
    mask_token_id: int,
    pad_token_id: int,
    max_sequence_length: int,
    probability: float,
    mean_span_length: float,
    whole_word_denoising: bool,
    decoder_start_token_id: int,
    max_number_of_spans: int = None,
) -> Tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]
]:
    r""" Denoising as in BART paper + whole word. """

    # generate a mask or spans spreaded over the whole sequence with
    # average length=mean_span_length and poisson distribution
    spans_mask = random_spans_mask(
        len(input_ids),
        probability=probability,
        mean_span_length=mean_span_length,
        max_noise_spans=max_number_of_spans,
        distribution='poisson',
    )

    # deactivate masks over special tokens
    spans_mask[word_ids == IGNORE_IDX] = False

    # correct mask to cover entire tokens and not just parts
    if whole_word_denoising:
        word_tails = whole_word_tails_mask(word_ids)
        spans_mask = correct_mask_to_cover_words(spans_mask, word_tails)

    # divide tokens between input_ids and labels and add special sentinel tokens
    labels = input_ids.copy()
    input_ids = compress_spans_to_unique_tokens(
        input_ids, mask=spans_mask, unique_tokens=mask_token_id
    )

    # encoder inputs
    input_ids = pad_array(input_ids, padding_value=pad_token_id, length=max_sequence_length, padding_side='right')
    attention_mask = generate_new_attention_mask(input_ids, pad_token_id)

    # decoder inputs and labels
    labels = pad_array(labels, padding_value=pad_token_id, length=max_sequence_length, padding_side='right')
    labels[labels == pad_token_id] = IGNORE_IDX
    decoder_input_ids = shift_tokens_right(
        labels, pad_token_id=pad_token_id, decoder_start_token_id=decoder_start_token_id
    )
    decoder_attention_mask = generate_new_attention_mask(decoder_input_ids, pad_token_id)

    return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels


def denoising_t5_version(
    input_ids: npt.NDArray[np.int64],
    word_ids: npt.NDArray[np.int64],
    additional_special_tokens_ids: List[int],
    pad_token_id: int,
    max_sequence_length: int,
    probability: float,
    mean_span_length: float,
    whole_word_denoising: bool,
    decoder_start_token_id: int,
    max_number_of_spans: int = None,
) -> Tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]
]:
    r""" Denoising as in T5 paper + whole word.
    Many functions has been translated from the originals in tf to numpy. """

    if max_number_of_spans is None:
        max_number_of_spans = len(additional_special_tokens_ids)
    else:
        max_number_of_spans = min(max_number_of_spans, len(additional_special_tokens_ids))

    # generate a mask or spans spreaded over the whole sequence with
    # average length=mean_span_length and exponential distribution
    spans_mask = random_spans_mask(
        len(input_ids),
        probability=probability,
        mean_span_length=mean_span_length,
        max_noise_spans=max_number_of_spans,
        distribution='exponential',
    )

    # deactivate masks over special tokens
    spans_mask[word_ids == IGNORE_IDX] = False

    # correct mask to cover entire tokens and not just parts
    if whole_word_denoising:
        word_tails = whole_word_tails_mask(word_ids)
        spans_mask = correct_mask_to_cover_words(spans_mask, word_tails)

    # divide tokens between input_ids and labels and add special sentinel tokens
    labels = compress_spans_to_unique_tokens(
        input_ids, mask=~spans_mask, unique_tokens=additional_special_tokens_ids
    )
    input_ids = compress_spans_to_unique_tokens(
        input_ids, mask=spans_mask, unique_tokens=additional_special_tokens_ids
    )

    # encoder inputs
    input_ids = pad_array(input_ids, padding_value=pad_token_id, length=max_sequence_length, padding_side='right')
    attention_mask = generate_new_attention_mask(input_ids, pad_token_id)

    # decoder inputs and labels
    labels = pad_array(labels, padding_value=pad_token_id, length=max_sequence_length, padding_side='right')
    labels[labels == pad_token_id] = IGNORE_IDX
    decoder_input_ids = shift_tokens_right(
        labels, pad_token_id=pad_token_id, decoder_start_token_id=decoder_start_token_id
    )
    decoder_attention_mask = generate_new_attention_mask(decoder_input_ids, pad_token_id)

    return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels

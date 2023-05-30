from typing import List

import numpy as np
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from transformers_framework.architectures.tokenization_utils import ExtendedTokenizerFast
from transformers_framework.utilities.processors import convert_word_ids


# ! If defaults are changed make sure to update all pipelines !
def advanced_tokenization(
    *data: List[str],
    tokenizer: PreTrainedTokenizerBase = None,
    max_sequence_length: int = None,
    truncation: str = 'longest_first',
    padding: str = 'max_length',
    add_special_tokens: bool = True,
    return_overflowing_tokens: bool = False,
    return_offsets_mapping: bool = False,
    return_attention_mask: bool = None,
    return_token_type_ids: bool = None,
    return_word_ids: bool = False,
    return_sequence_ids: bool = False,
    extended_token_type_ids: List[int] = None,
    return_dict: bool = True,
    is_split_into_words: bool = False,
    squeeze: bool = True,
    separated: bool = False,
) -> BatchEncoding:
    r""" Encode one or more sequences together. If length of data is greater than 2, an ExtendedTokenizer is required.

    Args:
        data: a list of strings to encode
        tokenizer: instance of huggingface tokenizer
        max_sequence_length: if provided, trucate to this sequence length
        truncation: the preferred truncation strategy
        padding: the preferred padding strategy
        add_special_tokens: whether to add special tokens when encoding
        return_overflowing_tokens: return tokens that were truncated
        return_offsets_mapping: return map between tokens and original words
        return_attention_mask: return the attention mask
        return_token_type_ids: return the token type ids
        return_word_ids: return the word ids converted to numpy array (None is IGNORE_IDX)
        return_sequence_ids: return the sequence ids converted to numpy array (None is IGNORE_IDX)
        return_dict: whether to return a simple dict or a batch encoding
        is_split_into_words: whether input text is already split into words
        squeeze: remove additional dimension from created numpy arrays
        separated: whether to encode every input text separated and then concatenate the results

    Returns:
        A dict of numpy arrays
    """

    if return_attention_mask is None:
        return_attention_mask = 'attention_mask' in tokenizer.model_input_names
    if return_token_type_ids is None:
        return_token_type_ids = 'token_type_ids' in tokenizer.model_input_names or extended_token_type_ids is not None

    tok_args = dict(
        add_special_tokens=add_special_tokens,
        return_attention_mask=return_attention_mask,
        return_token_type_ids=return_token_type_ids,
        max_length=max_sequence_length,
        truncation=truncation,
        padding=padding,
        return_overflowing_tokens=return_overflowing_tokens,
        return_offsets_mapping=return_offsets_mapping,
        return_tensors='np',
        is_split_into_words=is_split_into_words,
    )

    # make sure not to unpack every token
    if is_split_into_words:
        data = [data]

    if len(data) > 2 or separated:
        if not isinstance(tokenizer, ExtendedTokenizerFast):
            raise ValueError("Cannot encode more than 2 sequences without an ExtendedTokenizer")

        encoded = (tokenizer.encode_many_separated if separated else tokenizer.encode_many)(
            data, **tok_args, extended_token_type_ids=extended_token_type_ids
        )
    else:
        encoded = tokenizer(*data, **tok_args)

    if return_word_ids:
        encoded['word_ids'] = convert_word_ids(encoded.word_ids())
    if return_sequence_ids:
        encoded['sequence_ids'] = convert_word_ids(encoded.sequence_ids())

    # converting to simple dictionary if batch encoding functionalities are not needed
    if return_dict:
        encoded = dict(encoded)

    # remove additional dimensions
    if squeeze:
        for k in encoded.keys():
            if isinstance(encoded[k], np.ndarray) and encoded[k].ndim > 1:
                encoded[k] = encoded[k].squeeze()

    return encoded


def get_default_token_type_ids(token_ids: List[List[int]], extended_token_type_ids: List[int]) -> List[int]:
    r""" Get default sequence of sentence type ids. """

    if len(extended_token_type_ids) < len(token_ids):
        extended_token_type_ids = (
            extended_token_type_ids + [extended_token_type_ids[-1]] * (
                len(token_ids) - len(extended_token_type_ids)
            )
        )
    return extended_token_type_ids[:len(token_ids)]

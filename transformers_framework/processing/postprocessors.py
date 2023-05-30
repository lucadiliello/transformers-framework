from typing import Any, Dict, List

import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers_framework.language.modeling import (
    denoising_bart_version,
    denoising_t5_version,
    masked_language_modeling,
    random_token_substitution,
)
from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.arguments import extract_text_fields_with_multiple_formats
from transformers_framework.utilities.functional import shift_tokens_right
from transformers_framework.utilities.numpy import numpy_num_elements
from transformers_framework.utilities.processors import (
    generate_new_attention_mask,
    shuffle_sentences_util,
    word_to_token_labels,
)
from transformers_framework.utilities.tokenization import advanced_tokenization


def bart_denoising_processor(
    sample: Dict[str, Any],
    input_column: str,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    probability: float,
    mean_span_length: float,
    whole_word_denoising: bool,
    max_number_of_spans: int,
    shuffle_sentences: bool = False,
):
    r""" Tokenize text string and apply denoising BART version. """
    text = sample[input_column]
    if shuffle_sentences:
        text = shuffle_sentences_util(text)

    data = advanced_tokenization(
        text,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        padding=False,
        return_word_ids=True,
    )

    input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = denoising_bart_version(
        input_ids=data['input_ids'],
        word_ids=data['word_ids'],
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_sequence_length=max_sequence_length,
        probability=probability,
        mean_span_length=mean_span_length,
        whole_word_denoising=whole_word_denoising,
        decoder_start_token_id=tokenizer.decoder_start_token_id,
        max_number_of_spans=max_number_of_spans,
    )

    res = {}

    # encoder
    res['input_ids'] = input_ids
    res['attention_mask'] = attention_mask

    # decoder
    res['decoder_input_ids'] = decoder_input_ids
    res['decoder_attention_mask'] = decoder_attention_mask
    res['seq_to_seq_lm_labels'] = labels

    return res


def t5_denoising_processor(
    sample: Dict[str, Any],
    input_column: str,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    probability: float,
    mean_span_length: float,
    whole_word_denoising: bool,
    max_number_of_spans: int,
):
    r""" Tokenize text string and apply denoising T5 version. """
    data = advanced_tokenization(
        sample[input_column],
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        padding=False,
        return_word_ids=True,
    )

    input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = denoising_t5_version(
        input_ids=data['input_ids'],
        word_ids=data['word_ids'],
        additional_special_tokens_ids=tokenizer.additional_special_tokens_ids,
        pad_token_id=tokenizer.pad_token_id,
        max_sequence_length=max_sequence_length,
        probability=probability,
        mean_span_length=mean_span_length,
        whole_word_denoising=whole_word_denoising,
        decoder_start_token_id=tokenizer.decoder_start_token_id,
        max_number_of_spans=max_number_of_spans,
    )

    res = {}

    # encoder
    res['input_ids'] = input_ids
    res['attention_mask'] = attention_mask

    # decoder
    res['decoder_input_ids'] = decoder_input_ids
    res['decoder_attention_mask'] = decoder_attention_mask
    res['seq_to_seq_lm_labels'] = labels

    return res


def answer_selection_processor(
    sample: Dict[str, Any],
    input_columns: str,
    index_column: str,
    label_column: str,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    extended_token_type_ids: List[int] = None,
    k: int = 1,
):
    r""" Tokenize text string and prepare for AS2. """
    text = extract_text_fields_with_multiple_formats(sample, input_columns)

    res = advanced_tokenization(
        *text,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        extended_token_type_ids=extended_token_type_ids,
    )

    res['index'] = np.array(sample[index_column])
    res['seq_class_labels'] = np.array(sample[label_column])

    if k is not None:
        assert numpy_num_elements(res['index']) == k, f"expected {k}, got {numpy_num_elements(res['index'])}"  # nosec
        assert numpy_num_elements(res['seq_class_labels']) == k, (  # nosec
            f"expected {k}, got {numpy_num_elements(res['seq_class_labels'])}"
        )

    return res


def seq_class_processor(
    sample: Dict[str, Any],
    input_columns: List[str],
    label_column: str,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    extended_token_type_ids: List[int] = None,
    k: int = 1,
):
    r""" Tokenize text string and prepare for sequence classification. """
    text = extract_text_fields_with_multiple_formats(sample, input_columns)

    res = advanced_tokenization(
        *text,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        extended_token_type_ids=extended_token_type_ids,
    )

    res['seq_class_labels'] = np.array(sample[label_column])
    if k is not None:
        assert numpy_num_elements(res['seq_class_labels']) == k  # nosec

    return res


def question_answering_processor(
    sample: Dict[str, Any],
):
    r""" Convert model inputs to numpy arrays. """
    sample['index'] = np.array(sample['index'])

    # model inputs
    sample['input_ids'] = np.array(sample['input_ids'])
    if 'attention_mask' in sample:
        sample['attention_mask'] = np.array(sample['attention_mask'])
    if 'token_type_ids' in sample:
        sample['token_type_ids'] = np.array(sample['token_type_ids'])

    # labels
    sample['start_position_labels'] = np.array(sample.pop('start_positions'))
    sample['end_position_labels'] = np.array(sample.pop('end_positions'))

    return sample


def masked_lm_processor(
    sample: Dict[str, Any],
    input_columns: List[str],
    probability: float,
    probability_masked: float,
    probability_replaced: float,
    probability_unchanged: float,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    whole_word_masking: bool,
    return_original_input_ids: bool = False,
):
    r""" Tokenize text string and prepare for MLM. """
    assert 1 <= len(input_columns) <= 2, f"Allowed 1 or 2 inputs in `masked_lm_processor`, got {len(input_columns)}"

    text = [sample[input_column] for input_column in input_columns if sample[input_column] is not None]
    assert text

    data = advanced_tokenization(
        *text,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        return_word_ids=True,
    )

    input_ids, labels = masked_language_modeling(
        input_ids=data['input_ids'],
        word_ids=data.pop('word_ids'),
        mask_token_id=tokenizer.mask_token_id,
        vocab_size=tokenizer.vocab_size,
        probability=probability,
        specific_probabilities=(probability_masked, probability_replaced, probability_unchanged),
        whole_word_masking=whole_word_masking,
    )

    # encoder
    res = dict(data)
    res['input_ids'] = input_ids
    if return_original_input_ids:
        res['original_input_ids'] = data['input_ids']

    # labels
    res['masked_lm_labels'] = labels

    return res


def masked_lm_and_seq_class_processor(
    sample: Dict[str, Any],
    input_columns: List[str],
    label_column: str,
    probability: float,
    probability_masked: float,
    probability_replaced: float,
    probability_unchanged: float,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    whole_word_masking: bool,
    training: bool,
    extended_token_type_ids: List[int] = None,
    k: int = 1,
    return_original_input_ids: bool = False,
):
    r""" Tokenize text string and prepare for MLM + AS2. """
    text = extract_text_fields_with_multiple_formats(sample, input_columns)

    data = advanced_tokenization(
        *text,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        extended_token_type_ids=extended_token_type_ids,
        return_word_ids=True,
    )

    masked_input_ids, masked_lm_labels = masked_language_modeling(
        input_ids=data['input_ids'],
        word_ids=data.pop('word_ids'),
        mask_token_id=tokenizer.mask_token_id,
        vocab_size=tokenizer.vocab_size,
        probability=probability,
        specific_probabilities=(probability_masked, probability_replaced, probability_unchanged),
        whole_word_masking=whole_word_masking,
        disable=not training,
    )

    res = dict(data)

    # inputs
    res['input_ids'] = masked_input_ids
    if return_original_input_ids:
        res['original_input_ids'] = data['input_ids']

    # masked language modeling labels
    res['masked_lm_labels'] = masked_lm_labels

    # answer selection labels
    res['seq_class_labels'] = np.array(sample[label_column])
    if k is not None:
        assert numpy_num_elements(res['seq_class_labels']) == k  # nosec

    return res


def masked_lm_and_answer_selection_processor(
    sample: Dict[str, Any],
    input_columns: List[str],
    index_column: str,
    label_column: str,
    probability: float,
    probability_masked: float,
    probability_replaced: float,
    probability_unchanged: float,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    whole_word_masking: bool,
    training: bool,
    extended_token_type_ids: List[int] = None,
    k: int = 1,
):
    r""" Tokenize text string and prepare for MLM + AS2. """
    text = extract_text_fields_with_multiple_formats(sample, input_columns)

    data = advanced_tokenization(
        *text,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        extended_token_type_ids=extended_token_type_ids,
        return_word_ids=True,
    )

    masked_input_ids, masked_lm_labels = masked_language_modeling(
        input_ids=data.pop('input_ids'),
        word_ids=data.pop('word_ids'),
        mask_token_id=tokenizer.mask_token_id,
        vocab_size=tokenizer.vocab_size,
        probability=probability,
        specific_probabilities=(probability_masked, probability_replaced, probability_unchanged),
        whole_word_masking=whole_word_masking,
        disable=not training,
    )

    res = dict(data)

    # inputs
    res['input_ids'] = masked_input_ids

    # masked language modeling labels
    res['masked_lm_labels'] = masked_lm_labels

    # answer seletion labels
    res['index'] = np.array(sample[index_column])
    res['seq_class_labels'] = np.array(sample[label_column])

    if k is not None:
        assert numpy_num_elements(res['index']) == k  # nosec
        assert numpy_num_elements(res['seq_class_labels']) == k  # nosec

    return res


def masked_lm_and_token_class_processor(
    sample: Dict[str, Any],
    input_column: str,
    label_column: str,
    probability: float,
    probability_masked: float,
    probability_replaced: float,
    probability_unchanged: float,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    whole_word_masking: bool,
):
    r""" Tokenize text string and prepare for MLM + Sequence Classification. """
    data = advanced_tokenization(
        sample[input_column],
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        is_split_into_words=True,
        return_dict=False,
        return_word_ids=True,
    )

    # masked language modeling
    masked_input_ids, labels = masked_language_modeling(
        input_ids=data['input_ids'],
        word_ids=data.pop('word_ids'),
        mask_token_id=tokenizer.mask_token_id,
        vocab_size=tokenizer.vocab_size,
        probability=probability,
        specific_probabilities=(probability_masked, probability_replaced, probability_unchanged),
        whole_word_masking=whole_word_masking,
    )

    # model inputs
    res = dict(data)

    # encoder
    res['input_ids'] = masked_input_ids
    res['masked_lm_labels'] = labels

    # compute first token classification labels
    word_class_labels = sample[label_column]
    res['token_class_labels'] = word_to_token_labels(word_class_labels, data)

    return res


def summarization_processor(
    sample: Dict[str, Any],
    document_column: str,
    summary_column: str,
    prefix: str,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: List[int],
    additional_summaries_column: str = None,
):
    r""" Tokenize text string and prepare for Summarization. """
    # encoder
    res_document = advanced_tokenization(
        prefix + sample[document_column],
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length[0],
    )

    # decoder
    res_summary = advanced_tokenization(
        sample[summary_column],
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length[1],
    )

    # create labels from original summary input ids
    labels = res_summary['input_ids']
    labels[labels == tokenizer.pad_token_id] = IGNORE_IDX
    decoder_input_ids = shift_tokens_right(
        labels,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.decoder_start_token_id,
    )

    res = {}

    # encoder
    res['input_ids'] = res_document['input_ids']
    res['attention_mask'] = res_document['attention_mask']

    # decoder
    res['decoder_input_ids'] = decoder_input_ids
    res['decoder_attention_mask'] = generate_new_attention_mask(
        decoder_input_ids, pad_token_id=tokenizer.pad_token_id
    )

    res['seq_to_seq_lm_labels'] = labels
    res['original_summary'] = sample[summary_column]
    if isinstance(res['original_summary'], str):
        res['original_summary'] = [res['original_summary']]
    if additional_summaries_column is not None and additional_summaries_column in sample:
        res['original_summary'] += sample[additional_summaries_column]

    return res


def token_class_processor(
    sample: Dict[str, Any],
    input_column: str,
    label_column: str,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: List[int],
):
    r""" Tokenize text string and prepare for Token Classification. """
    data = advanced_tokenization(
        sample[input_column],
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        is_split_into_words=True,
        return_dict=False,
    )

    # model inputs
    res = dict(data)

    # encoder
    labels = sample[label_column]
    res['token_class_labels'] = word_to_token_labels(labels, data)

    return res


def random_token_detection_processor(
    sample: Dict[str, Any],
    input_column: str,
    probability: float,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    whole_word_detection: bool,
):
    r""" Tokenize text string and prepare for RTD. """
    data = advanced_tokenization(
        sample[input_column],
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        return_word_ids=True,
    )

    input_ids, labels = random_token_substitution(
        input_ids=data.pop('input_ids'),
        word_ids=data.pop('word_ids'),
        vocab_size=tokenizer.vocab_size,
        probability=probability,
        whole_word_detection=whole_word_detection,
    )

    # encoder
    res = dict(data)
    res['input_ids'] = input_ids
    res['token_detection_labels'] = labels
    return res

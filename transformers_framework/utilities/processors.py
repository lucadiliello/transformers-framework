import re
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from numba import njit
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.functional import check_is_max_context, dict2list, list2dict
from transformers_framework.utilities.logging import rank_zero_error


def clean_text_for_qa(
    tokenizer: PreTrainedTokenizerBase, text: str, spans: List[Tuple] = None
) -> Union[str, Tuple[str, List]]:
    r""" Clean a string and return mapping to old string. Also fixes eventual spans referring te the old string. """
    special_tokens_to_remove = set(['[DOC]', '[PAR]', '[TLE]', '[SEP]'] + tokenizer.all_special_tokens)
    regex = re.compile(
        r"(\s+)|(" + "|".join(re.escape(x) for x in special_tokens_to_remove) + ")"
    )

    # for each match, remove it from context and adjust spans position
    old_char_to_new_char = {}
    new_text = ""
    old_index = 0

    while old_index < len(text):
        match = regex.search(text, old_index)
        if match is None:
            # add last mapping to new_char_to_old_char
            old_char_to_new_char.update({old_index + i: len(new_text) + i for i in range(len(text) - old_index)})
            new_text += text[old_index:]
            break

        # advance copying old in new
        span = match.span()
        old_char_to_new_char.update({old_index + i: len(new_text) + i for i in range(span[0] - old_index)})
        new_text += text[old_index:span[0]]
        old_index = span[0]

        # now work on match
        if match.group()[0] is not None:
            # stripping whitespaces
            if not new_text.endswith(" ") and len(new_text):
                new_text += " "
        else:
            # removing special reserved tokens
            pass

        old_index = span[1]

    new_text = new_text.strip()

    if spans is not None:
        # fix position of spans after cleaning
        spans = [
            (old_char_to_new_char[span_start], old_char_to_new_char[span_end])
            for span_start, span_end in spans
            if span_start in old_char_to_new_char and span_end in old_char_to_new_char
        ]
        return new_text, spans
    else:
        return new_text


@njit
def whole_word_tails_mask(word_ids: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    r""" Create whole work masking mask.
    IGNORE_IDX for special tokens, 0 for heads and 1 for tails. """
    res = np.zeros_like(word_ids, dtype=np.int64)
    res[0] = IGNORE_IDX if (word_ids[0] == IGNORE_IDX) else 0

    for i in range(1, len(word_ids)):
        res[i] = (word_ids[i - 1] == word_ids[i]) if word_ids[i] != IGNORE_IDX else IGNORE_IDX
    return res


@njit
def advance_on_input_specials(word_tails: npt.NDArray[np.int64], position: int) -> int:
    r""" Advance position over the input until a non special token is found.

    Returns:
        position of token next to the limit for easy indexing/slicing.
    """

    while position < len(word_tails) and word_tails[position] == IGNORE_IDX:
        position += 1

    return position


@njit
def advance_on_inputs_tokens(
    word_tails: npt.NDArray[np.int64], position: int, span_length: int, whole_word_denoising: bool = False
) -> int:
    r""" Advance position over the input taking into account the span_lenght
    and whether we are working with words or tokens.

    Returns:
        position of token next to the limit for easy indexing/slicing.
    """

    if whole_word_denoising:
        assert word_tails[position] == 0, (  # nosec
            "cannot start advancing from non head token when whole_word_denoising is True"
        )

    j = 0  # counter for number of tokens or words passed
    while j < span_length:
        position += 1  # we work on the next token at each iteration

        # if we reached end of input
        if position + 1 >= len(word_tails):
            return position

        # if we found a special token
        if word_tails[position] == IGNORE_IDX:
            return position

        # if we found a head token
        if word_tails[position] == 0:
            j += 1

        # if we found a tail token
        if word_tails[position] == 1:
            if not whole_word_denoising:
                j += 1

    return position


def generate_new_attention_mask(input_ids: List[int], pad_token_id: int) -> List[int]:
    r""" Generate attention mask for new inputs. """
    if isinstance(input_ids, np.ndarray):
        return (input_ids != pad_token_id).astype(np.int64)
    return [int(x != pad_token_id) for x in input_ids]


def get_arch_sentence_splitter() -> Callable:
    r""" Return a sentence splitter fn based on architecture because not all packages are avaialable in arm64. """

    try:
        from blingfire import text_to_sentences

        def sentence_splitter(string: str) -> List[str]:
            return text_to_sentences(string).split("\n")

    except (ModuleNotFoundError):
        rank_zero_error("Module `blingfire` not found. Install it with `pip install blingfire`")
        exit(1)

    except (ImportError, OSError):  # blingfire seems not to work on arm64 architectures: we default back to nltk
        try:
            import nltk

            def sentence_splitter(string: str) -> List[str]:
                return nltk.sent_tokenize(string)

        except ModuleNotFoundError:
            rank_zero_error("Module `nltk` not found. Install it with `pip install nltk`")
            exit(1)

    return sentence_splitter


sentence_splitter = get_arch_sentence_splitter()


def shuffle_sentences_util(text: str) -> str:
    r""" Shuffle the sentences in text. """
    sentences = sentence_splitter(text)
    return " ".join(sentences)


def pad_or_truncate_encoded_sequence(sequence: List[int], tokenizer: PreTrainedTokenizerBase, max_length: int):
    r""" Encode again already encoded sequence, for automatic truncation, padding and special tokens management. """

    # padding
    if len(sequence) <= max_length:
        sequence = sequence + [tokenizer.pad_token_id] * (max_length - len(sequence))

    # truncation
    else:
        sequence = sequence[:max_length]

        # make sure sequence still ends with eos token
        for i in range(len(sequence))[::-1]:
            if sequence[i] != tokenizer.pad_token_id:
                if sequence[i] != tokenizer.eos_token_id:
                    sequence[i] = tokenizer.eos_token_id
                    break

    return sequence


def convert_word_ids(word_ids: List[Union[int, None]]) -> npt.NDArray[np.int64]:
    r""" Convert word_ids setting None to IGNORE_IDX. """
    res = np.array(word_ids)
    res[res == None] = IGNORE_IDX  # noqa: E711
    return res.astype(np.int64)


@njit
def extend_mask_on_tails(
    masked_indices: npt.NDArray[np.bool_], word_tails: npt.NDArray[np.bool_]
) -> npt.NDArray[np.bool_]:
    r""" Extend mask over all tails if head is masked. """
    word_tails = (word_tails == 1)

    res = np.zeros_like(masked_indices)
    for i in range(1, len(masked_indices)):
        res[i] = masked_indices[i] | (masked_indices[i - 1] & word_tails[i])
    return res


def word_to_token_labels(
    labels: npt.NDArray[np.int64], encoding: BatchEncoding, label_only_on_first_token: bool = True
) -> npt.NDArray[np.int64]:
    r""" Convert labels over words to labels over tokens. """
    res = np.full_like(encoding.input_ids, fill_value=IGNORE_IDX)
    for i, label in enumerate(labels):
        position = encoding.word_to_tokens(i)
        if position is not None:
            if label_only_on_first_token:
                res[position.start] = label
            else:
                res[position.start:position.end] = label
    return res


def process_entry_question_answering(
    entry: Dict,
    index: int,
    tokenizer: PreTrainedTokenizerBase = None,
    query_column: str = None,
    context_column: str = None,
    answers_column: str = None,
    label_column: str = None,
) -> Dict:
    r""" Process and clean a single dataset entry. """
    # get all spans in order and clean data
    char_spans = sorted(set([
        (start, end)
        for span in entry[label_column]
        for start, end in zip(span['start'], span['end'])
    ]))

    # map old tokens to new, skipping unwanted junk. remap also gold spans
    context = entry[context_column]

    # all answers
    all_extracted_answers = [context[char_start:char_end + 1] for char_start, char_end in char_spans]

    # take first span for training
    if char_spans:
        selected_span = char_spans[0]
        answer = all_extracted_answers[0]
        char_start_position, char_end_position = selected_span
    else:
        answer = char_start_position = char_end_position = None

    # remove eventual multiple spaces
    question = clean_text_for_qa(tokenizer, entry[query_column])

    example = dict(
        index=index,
        context=context,
        answer=answer,
        gold_answers=set(entry[answers_column]),
        char_start_position=char_start_position,
        char_end_position=char_end_position,
        question_text=question,
    )
 
    return example


def convert_examples_to_features(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase = None,
    max_sequence_length: int = None,
    doc_stride: int = None,
    max_query_length: int = None,
) -> Dict[str, List]:
    r""" Convert samples with annotations in one or more tokenized features. """
    examples = dict2list(examples)
    examples = [
        res
        for sample in examples
        for res in _convert_example_to_features(
            sample,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
        )
    ]
    return list2dict(examples)


def _convert_example_to_features(
    example: Dict,
    tokenizer: PreTrainedTokenizerBase = None,
    max_sequence_length: int = None,
    doc_stride: int = None,
    max_query_length: int = None,
) -> Dict[str, List]:
    r""" Convert a sample with annotations in one or more tokenized features. """
    # clip question to max length in tokens with this trick
    question_text = tokenizer.convert_tokens_to_string(
        tokenizer.tokenize(example['question_text'], add_special_tokens=False)[:max_query_length]
    )

    # for all tokenizers compatibility
    encoded = tokenizer(
        question_text,
        example['context'],
        truncation='only_second',
        padding="max_length",
        max_length=max_sequence_length,
        stride=doc_stride,
        add_special_tokens=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    num_features = len(encoded['input_ids'])
    all_sequence_ids = [encoded.sequence_ids(i) for i in range(num_features)]

    # map: token position in the span -> bool (is max context)
    token_is_max_context = list(check_is_max_context(all_sequence_ids, doc_stride=doc_stride))

    for feature_index, input_ids in enumerate(encoded['input_ids']):

        tok_start_position = 0
        tok_end_position = 0
    
        # convert positions from words relativity to tokens
        if example['char_start_position'] is not None and example['char_end_position'] is not None:
            start_position = encoded.char_to_token(feature_index, example['char_start_position'], 1)
            end_position = encoded.char_to_token(feature_index, example['char_end_position'], 1)

            # if gold answer is contained in actual span
            if start_position is not None and end_position is not None:
                tok_start_position = start_position
                tok_end_position = end_position

        # offset mapping only on context
        covered_tokens = [i for i, seq_id in enumerate(all_sequence_ids[feature_index]) if seq_id == 1]

        res = dict(
            index=example['index'],
            context=example['context'],
            tokens=encoded.tokens(feature_index),
            covered_tokens=covered_tokens,
            offset_mapping=encoded['offset_mapping'][feature_index],
            token_is_max_context=list(token_is_max_context[feature_index].items()),
            input_ids=input_ids,
            start_positions=tok_start_position,
            end_positions=tok_end_position,
            gold_answers=example['gold_answers'],
        )

        if 'attention_mask' in encoded:
            res['attention_mask'] = encoded['attention_mask'][feature_index]
        if 'token_type_ids' in encoded:
            res['token_type_ids'] = encoded['token_type_ids'][feature_index]

        yield res

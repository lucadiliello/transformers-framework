import types
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from transformers import PreTrainedTokenizerFast
from transformers.file_utils import PaddingStrategy, TensorType
from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
    logger,
)

from transformers_framework.utilities.functional import special_zip


class ExtendedTokenizerFast(PreTrainedTokenizerFast, ABC):

    def _get_input_ids(
        self,
        text: Union[List[int], str],
        return_offsets_mapping: bool = False,
        return_word_ids: bool = False,
        **kwargs,
    ) -> Tuple[List[int], List[Tuple[int, int]], List[int]]:
        r""" Given a string or a list of ids, return tuple of ids and offset map. """
        if isinstance(text, str):
            encoding = self.encode_plus(
                text, **kwargs, add_special_tokens=False, return_offsets_mapping=return_offsets_mapping
            )
            return (
                self.convert_tokens_to_ids(encoding.tokens()),
                encoding['offset_mapping'] if return_offsets_mapping else None,
                encoding.word_ids() if return_word_ids else None,
            )
        elif isinstance(text, (list, tuple)) and (len(text) == 0 or (len(text) > 0 and isinstance(text[0], int))):
            if return_offsets_mapping is True:
                raise ValueError("To return offset map, you must tokenize strings and not lists of ids.")
            if return_word_ids is True:
                raise ValueError("To return word ids, you must tokenize strings and not lists of ids.")
            return (text, None, None)
        else:
            raise ValueError(
                f"Input {text} is not valid. Should be a string, a list/tuple "
                f"of strings or a list/tuple of integers."
            )

    def encode_many(
        self,
        texts: Union[TextInput, PreTokenizedInput, EncodedInput],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        extended_token_type_ids: int = None,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        r"""
        Tokenize and prepare for the model many consecutive sequences.

        Args:
            texts (:obj:`str`, :obj:`List[str]` or `List[List[int]]`:
                The sequences to be encoded together. This should be a list of strings or a list of integers
                (tokenized string ids using the ``convert_tokens_to_ids`` method).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        input_ids, offsets_map, word_ids = list(zip(*[
            self._get_input_ids(
                text, return_offsets_mapping=return_offsets_mapping, return_word_ids=True, **kwargs
            ) for text in texts
        ]))
        # need lists and not tuples
        input_ids = list(input_ids)
        word_ids = list(word_ids)

        if not return_offsets_mapping:
            offsets_map = None  # was a list of None
        else:
            offsets_map = list(offsets_map)

        return self.prepare_for_model_many(
            input_ids,
            offsets_map=offsets_map,
            word_ids=word_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            extended_token_type_ids=extended_token_type_ids,
        )

    def prepare_for_model_many(
        self,
        input_ids: List[List[int]],
        offsets_map: List[List[Tuple[int, int]]] = None,
        word_ids: List[List[int]] = None,
        add_special_tokens: bool = True,
        padding_strategy: Union[bool, str, PaddingStrategy] = False,
        truncation_strategy: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        extended_token_type_ids: int = None,
    ) -> BatchEncoding:
        r"""
        Prepares a tuple of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens.

        Args:
            input_ids (:obj:`List[List[int]]`):
                Tokenized input ids of the sequences. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
        """

        if truncation_strategy in (TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND):
            raise ValueError(
                f"truncation_strategy must be set to {TruncationStrategy.LONGEST_FIRST} or "
                f"{TruncationStrategy.DO_NOT_TRUNCATE} (got {truncation_strategy})"
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = self.get_encoding_length(input_ids, add_special_tokens=add_special_tokens)

        # Fix max_length if not supplied by user
        if not max_length:
            if self.model_max_length is not None:
                logger.warning(
                    f"Since not max_length was supplied for truncation, using model_max_length={self.model_max_length}"
                )
                max_length = self.model_max_length
            else:
                raise ValueError("Cannot truncate without setting `max_length` because model has no `model_max_length`")

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and total_len > max_length:
            input_ids, offsets_map, word_ids, overflowing_tokens = self.truncate_many_sequences(
                input_ids,
                offsets_map,
                word_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        sequence = self.build_many_inputs_with_special_tokens(input_ids, add_special_tokens=add_special_tokens)

        # build offsets_map from the encoding of multiple sequences
        if offsets_map is not None:
            offsets_map = self.build_many_offsets_with_special_tokens(
                offsets_map, add_special_tokens=add_special_tokens
            )

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            token_type_ids = self.create_token_type_ids_from_many_sequences(
                input_ids, add_special_tokens=add_special_tokens, extended_token_type_ids=extended_token_type_ids
            )
            encoded_inputs["token_type_ids"] = token_type_ids

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if self.padding_side != 'right':
            raise ValueError("Extended tokenizers do not work with left padding yet")

        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        # compute anyway because it will be used below
        special_tokens_mask = self.get_special_tokens_mask(encoded_inputs["input_ids"], already_has_special_tokens=True)
        if return_special_tokens_mask:
            encoded_inputs["special_tokens_mask"] = special_tokens_mask

        # Eventually pad offset map
        if return_offsets_mapping and offsets_map is not None and len(offsets_map) < len(encoded_inputs["input_ids"]):
            offsets_map += [(0, 0)] * (len(encoded_inputs["input_ids"]) - len(offsets_map))
            encoded_inputs['offset_mapping'] = offsets_map

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        # build sequence ids even if this is not a fast tokenizer
        sequence_ids = self.create_sequence_ids_from_many_sequences(
            input_ids, special_tokens_mask, add_special_tokens=add_special_tokens
        )

        # build word ids even if this is not a fast tokenizer
        if word_ids is not None:
            # this ensures different texts has no overlapping sequence ids
            word_ids = fix_word_ids(word_ids)

            # the following lines add Nones to the word ids when there is a special token
            word_ids = self.build_many_inputs_with_special_tokens(word_ids, add_special_tokens=add_special_tokens)
            word_ids = [None if special else w for w, special in zip(word_ids, special_tokens_mask)]

            # pad word ids to match input ids length
            if len(word_ids) < len(encoded_inputs["input_ids"]):
                word_ids += [None] * (len(encoded_inputs["input_ids"]) - len(word_ids))

        # this BatchEncoding object will be slow tokenizers like
        res = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        # we need to manually enable sequence ids in BatchEncoding because is not enabled by default in slow tokenizers
        add_sequence_ids_to_batch_encoding(res, sequence_ids)

        # we need to manually enable word ids in BatchEncoding because is not enabled by default in slow tokenizers
        add_word_ids_to_batch_encoding(res, word_ids)

        return res

    def truncate_many_sequences(
        self,
        input_ids: List[List[int]],
        offsets_map: List[List[Tuple[int, int]]] = None,
        word_ids: List[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[List[int]], List[List[Tuple[int, int]]], List[List[int]], ]:
        r"""
        Truncates a sequence pair in-place following the strategy.

        Args:
            input_ids (:obj:`List[List[int]]`):
                Tokenized input ids of the sequences. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
            offsets_map (:obj:`List[List[Tuple[int, int]]]`): list of offsets maps for every input
            num_tokens_to_remove (:obj:`int`, `optional`, defaults to 0):
                Number of tokens to remove using the truncation strategy.
            truncation_strategy (:obj:`str` or :class:`~transformers.tokenization_utils_base.TruncationStrategy`,
                `optional`, defaults to :obj:`False`):
                The strategy to follow for truncation. Can be:

                * :obj:`'longest_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate token by token, removing a token from the longest sequence in the pair if a pair of
                  sequences (or a batch of pairs) is provided.
                * :obj:`'only_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or to
                  the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_second'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
            stride (:obj:`int`, `optional`, defaults to 0):
                If set to a positive number, the overflowing tokens returned will contain some tokens from the main
                sequence returned. The value of this argument defines the number of additional tokens.

        Returns:
            :obj:`Tuple[List[List[int]], List[List[Tuple[int, int]]], List[List[int]], List[int]`: The truncated
            ``input_ids`` and ``offsets_map`` and ``word_ids`` and the list of overflowing tokens.
        """

        if num_tokens_to_remove <= 0:
            return input_ids, offsets_map, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        # checks over lengths
        if offsets_map is not None:
            if len(offsets_map) != len(input_ids):
                raise ValueError(
                    f"Got `offset_map` with length {len(offsets_map)} and `input_ids` with length {len(input_ids)}"
                )
            for offset_map, inputs in zip(offsets_map, input_ids):
                if len(offset_map) != len(inputs):
                    raise ValueError(
                        f"Got `offset_map` with length {len(offsets_map)} and `input_ids` with length {len(input_ids)}"
                    )

        if word_ids is not None:
            if len(word_ids) != len(input_ids):
                raise ValueError(
                    f"Got `word_ids` with length {len(word_ids)} and `input_ids` with length {len(input_ids)}"
                )
            for word_id, inputs in zip(word_ids, input_ids):
                if len(word_id) != len(inputs):
                    raise ValueError(
                        f"Got `word_ids` with length {len(word_id)} and `input_ids` with length {len(input_ids)}"
                    )

        overflowing_tokens = None
        if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            lengths = [len(i) for i in input_ids]

            for _ in range(num_tokens_to_remove):
                # Get the length of the longer sequence in ids
                longer_index = lengths.index(max(lengths))
                lengths[longer_index] -= 1

            # shrink every input sequence to the length computed before
            overflowing_tokens = []
            for i in range(len(input_ids)):
                if len(input_ids[i]) > lengths[i]:

                    # compute overflowing tokens with stride
                    window_len = min(len(input_ids[i]), stride)
                    overflowing_tokens.append(input_ids[i][lengths[i] - window_len:])

                    # reduce lists
                    input_ids[i] = input_ids[i][:lengths[i]]

                    if offsets_map is not None:
                        offsets_map[i] = offsets_map[i][:lengths[i]]
                    if word_ids is not None:
                        word_ids[i] = word_ids[i][:lengths[i]]

        return input_ids, offsets_map, word_ids, overflowing_tokens

    def create_sequence_ids_from_many_sequences(
        self, token_ids: List[List[int]], special_tokens_mask: List[int], add_special_tokens: bool = True
    ) -> List[int]:
        r""" Create list of sequence ids automatically starting from token type ids and special tokens mask. """
        extended_token_type_ids = self.create_token_type_ids_from_many_sequences(
            token_ids, add_special_tokens=add_special_tokens, extended_token_type_ids=len(token_ids)
        )
        return [
            token_type if not special_token else None
            for token_type, special_token in special_zip(extended_token_type_ids, special_tokens_mask, stop='longest')
        ]

    @abstractmethod
    def build_many_inputs_with_special_tokens(
        self, token_ids: List[List[int]], add_special_tokens: bool = True
    ) -> List[int]:
        r"""
        Build model inputs from a tuple of sequences for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - tuple of sequences: ``[CLS] A [SEP] B [SEP] ... [SEP]``

        Args:
            token_id (:obj:`List[List[int]]`):
                List of IDs to which the special tokens will be added.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__
            with the appropriate special tokens.
        """

    @abstractmethod
    def build_many_offsets_with_special_tokens(
        self, offsets_map: List[List[Tuple[int, int]]], add_special_tokens: bool = True
    ) -> List[Tuple[int, int]]:
        r"""
        Build model offsets from a list of lists of positions and eventually add special tokenss.
        Positions for special tokens will be (0, 0). This methods is usually
        very similar to `build_many_inputs_with_special_tokens` but with different padding values.

        Args:
            offsets_map (:obj:`List[List[Tuple[int, int]]]`):
                List of offsets for each input piece.

        Returns:
            :obj:`List[Tuple[int, int]]`: List of offset maps with the appropriate special tokens.
        """

    @abstractmethod
    def create_token_type_ids_from_many_sequences(
        self,
        token_ids: List[List[int]],
        add_special_tokens: bool = True,
        extended_token_type_ids: Union[int, List[int]] = None,
    ) -> List[int]:
        r"""
        Create a mask from the two sequences passed to be used in a sequences classification task. A BERT sequences
        mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence | third sequence | .... |

        Args:
            token_ids (:obj:`List[List[int]]`):
                List of list of IDs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """

    @abstractmethod
    def get_encoding_length(self, token_ids: List[List[int]], add_special_tokens: bool = True) -> int:
        r"""
        Get the length of the encoding.

        Args:
            token_ids (:obj:`List[List[int]]`):
                List of list of IDs.

        Returns:
            :obj:`int`: Length of the encoding.
        """


def add_sequence_ids_to_batch_encoding(batch_encoding: BatchEncoding, sequence_ids: List[int]):
    r""" Add possibility to return `sequence_ids` to non fast tokenizer. """
    def sequence_ids_fn(self):
        return sequence_ids

    batch_encoding.sequence_ids = types.MethodType(sequence_ids_fn, batch_encoding)


def add_word_ids_to_batch_encoding(batch_encoding: BatchEncoding, word_ids: List[int]):
    r""" Add possibility to return `sequence_ids` to non fast tokenizer. """
    def word_ids_fn(self):
        return word_ids

    batch_encoding.word_ids = types.MethodType(word_ids_fn, batch_encoding)


def fix_word_ids(word_ids: List[List[int]]) -> List[int]:
    r""" Fix word ids such that every sublists starts from the id following the last in the previous list. """
    def get_highest_word_id(seq: List[int]):
        arg = [v for v in seq if v is not None]
        return max(arg) - min(arg) if arg else 0
    res = []
    actual_max = 0
    for part in word_ids:
        res.append([p + actual_max if p is not None else None for p in part])
        actual_max += get_highest_word_id(part) + 1
    return res

from typing import List, Tuple

from transformers.models.deberta_v2 import DebertaV2TokenizerFast

from transformers_framework.architectures.tokenization_utils import ExtendedTokenizerFast
from transformers_framework.utilities.tokenization import get_default_token_type_ids


class DebertaV2ExtendedTokenizerFast(DebertaV2TokenizerFast, ExtendedTokenizerFast):

    def build_many_inputs_with_special_tokens(
        self, token_ids: List[List[int]], add_special_tokens: bool = True
    ) -> List[int]:
        r"""
        Build model inputs from a tuple of sequences for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTA V2/V3 sequence has the following format:

        - tuple of sequences: ``[CLS] A [SEP] B [SEP] ... [SEP]``

        Args:
            token_id (:obj:`List[List[int]]`):
                List of IDs to which the special tokens will be added.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__
            with the appropriate special tokens.
        """
        if not add_special_tokens:
            return sum(token_ids, [])

        if len(token_ids) == 0:
            return [self.cls_token_id, self.sep_token_id]
        else:
            return [self.cls_token_id] + sum([ids + [self.sep_token_id] for ids in token_ids], [])

    def build_many_offsets_with_special_tokens(
        self, offsets_map: List[List[Tuple[int, int]]], add_special_tokens: bool = True
    ) -> List[Tuple[int, int]]:
        r"""
        Build model offsets from a list of lists of positions and eventually add special tokens.
        Positions for special tokens will be (0, 0). This methods is usually
        very similar to `build_many_inputs_with_special_tokens` but with different padding values.

        Args:
            offsets_map (:obj:`List[List[Tuple[int, int]]]`):
                List of offsets for each input piece.

        Returns:
            :obj:`List[Tuple[int, int]]`: List of offset maps with the appropriate special tokens.
        """
        if not add_special_tokens:
            return sum(offsets_map, [])

        if len(offsets_map) == 0:
            return [(0, 0), (0, 0)]
        else:
            return [(0, 0)] + sum([offset_map + [(0, 0)] for offset_map in offsets_map], [])

    def create_token_type_ids_from_many_sequences(
        self,
        token_ids: List[List[int]],
        add_special_tokens: bool = True,
        extended_token_type_ids: List[int] = None,
    ) -> List[int]:
        r"""
        Create a mask from the two sequences passed to be used in a sequences classification task.
        A DeBERTA V2/V3 sequences mask has the following format:

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

        if extended_token_type_ids is None:  # DeBERTA V2/V3 allows 2 token types by default
            extended_token_type_ids = [0, 1]

        sentence_ids = get_default_token_type_ids(token_ids, extended_token_type_ids)

        if not add_special_tokens:
            res = [s for s, ids in zip(sentence_ids, token_ids) for _ in ids]
        elif len(token_ids) == 0:
            res = [sentence_ids[0]] * 2
        else:
            res = [sentence_ids[0]] * (len(token_ids[0]) + 2) + [
                s for s, ids in zip(sentence_ids[1:], token_ids[1:]) for _ in range(len(ids) + 1)
            ]

        return res

    def get_encoding_length(self, token_ids: List[List[int]], add_special_tokens: bool = True) -> int:
        r"""
        Get the length of the encoding.

        Args:
            token_ids (:obj:`List[List[int]]`):
                List of list of IDs.

        Returns:
            :obj:`int`: Length of the encoding.
        """
        if not add_special_tokens:
            return sum(len(ids) for ids in token_ids)

        if not token_ids:
            return 2
        else:
            return sum(len(ids) + 1 for ids in token_ids) + 1

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers.modeling_outputs import BaseModelOutput


@dataclass
class MaskedLMOutput(BaseModelOutput):
    r"""
    Base class for masked language modeling outputs.

    Args:
        masked_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`,
            `optional`, returned when :obj:`masked_lm_labels` is provided): Masked language modeling (MLM) loss.
        masked_lm_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """

    masked_lm_loss: Optional[torch.FloatTensor] = None
    masked_lm_logits: Optional[torch.FloatTensor] = None


@dataclass
class SeqToSeqLMOutput(BaseModelOutput):
    r"""
    Base class for sequence-to-sequence language models outputs.

    Args:
        seq_to_seq_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`,
            `optional`, returned when :obj:`masked_lm_labels` is provided): Masked language modeling (MLM) loss.
        seq_to_seq_lm_logits
        (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states
        (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_last_hidden_state (`torch.FloatTensor`, *optional*):
            Hidden-states of the decoder at the output of each the last layer.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        decoder_labels (`torch.FloatTensor`, *optional*):
            `torch.FloatTensor` of shape `(batch_size, sequence_length)`.

            This may have a different shape in cases where training is speed up by computing lm head only on active
            labels.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    seq_to_seq_lm_loss: Optional[torch.FloatTensor] = None
    seq_to_seq_lm_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_labels: Optional[torch.Tensor] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class TokenDetectionOutput(BaseModelOutput):
    r"""
    Base class for token detection language modeling outputs.

    Args:
        token_detection_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`,
            `optional`, returned when :obj:`masked_lm_labels` is provided): Masked language modeling (MLM) loss.
        token_detection_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, 2)`):
            Prediction scores of the language modeling head (scores for each output token before SoftMax).
    """

    token_detection_loss: Optional[torch.FloatTensor] = None
    token_detection_logits: Optional[torch.FloatTensor] = None


@dataclass
class SeqClassOutput(BaseModelOutput):
    r"""
    Base class for sequence classification outputs.

    Args:
        seq_class_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`,
        `optional`, returned when :obj:`seq_class_labels` is provided):
            Sequence classification loss.
        seq_class_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Sequence classification logits (scores for each input example).
    """

    seq_class_loss: Optional[torch.FloatTensor] = None
    seq_class_logits: Optional[torch.FloatTensor] = None


class EmbeddingOutput(BaseModelOutput):
    r"""
    Base class for embedding outputs. Identical to BaseModelOutput.
    """


@dataclass
class TokenClassOutput(BaseModelOutput):
    r"""
    Base class for token classification outputs.

    Args:
        token_class_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`,
        `optional`, returned when :obj:`token_class_labels` is provided):
            Token classification loss.
        tok_class_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Token classification logits (scores for each token of each each input example).
    """

    token_class_loss: Optional[torch.FloatTensor] = None
    token_class_logits: Optional[torch.FloatTensor] = None


@dataclass
class QuestionAnsweringOutput(BaseModelOutput):
    r"""
    Base class for question answering outputs.

    Args:
        question_answering_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`,
        `optional`, returned when :obj:`start_position` and `end_position` is provided):
            Question answering loss.
        start_position_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, max_sequence_length, 2)`):
            QA classification logits for start position.
        end_position_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, max_sequence_length, 2)`):
            QA classification logits for end position.
    """

    question_answering_loss: Optional[torch.FloatTensor] = None
    start_position_logits: Optional[torch.FloatTensor] = None
    end_position_logits: Optional[torch.FloatTensor] = None


@dataclass
class MaskedLMAndSeqClassOutput(
    MaskedLMOutput, SeqClassOutput
):
    r""" Masked language modeling + sequence classification. """


@dataclass
class MaskedLMAndTokenClassOutput(
    MaskedLMOutput, TokenClassOutput
):
    r""" Masked language modeling + token classification. """


@dataclass
class TokenDetectionAndSeqClassOutput(
    TokenDetectionOutput, SeqClassOutput
):
    r""" Token detection + sequence classification. """


@dataclass
class MaskedLMAndQuestionAnsweringOutput(
    MaskedLMOutput, QuestionAnsweringOutput
):
    r""" Masked language modeling + question answering."""


@dataclass
class TokenDetectionAndQuestionAnsweringOutput(
    TokenDetectionOutput, QuestionAnsweringOutput
):
    r""" Token detection + question answering. """

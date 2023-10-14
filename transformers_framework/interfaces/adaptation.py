from typing import Dict

import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import MaskedLMOutput as TransformersMaskedLMOutput
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput as TransformersSeq2SeqLMOutput
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from transformers.models.electra.modeling_electra import ElectraForPreTrainingOutput

from transformers_framework.architectures.modeling_outputs import (
    EmbeddingOutput,
    MaskedLMAndTokenClassOutput,
    MaskedLMOutput,
    QuestionAnsweringOutput,
    SeqClassOutput,
    SeqToSeqLMOutput,
    TokenClassOutput,
    TokenDetectionOutput,
)


ADMISSIBLE_LABEL_NAMES = (
    'masked_lm_labels',
    'seq_class_labels',
    'token_class_labels',
    'token_detection_labels',
    'seq_to_seq_lm_labels',
    'start_position_labels',
    'end_position_labels',
)


def sequence_classification_adaptation(res: SequenceClassifierOutput) -> SeqClassOutput:
    r""" Convert transformers SequenceClassifierOutput to our SeqClassOutput. """
    if isinstance(res, SeqClassOutput):
        return res

    return SeqClassOutput(
        attentions=res.attentions,
        hidden_states=res.hidden_states,
        seq_class_loss=res.loss,
        seq_class_logits=res.logits,
    )


def retrieval_adaptation(res: BaseModelOutput) -> EmbeddingOutput:
    r""" Convert transformers SequenceClassifierOutput to our SeqClassOutput. """
    if isinstance(res, EmbeddingOutput):
        return res

    return EmbeddingOutput(
        attentions=res.attentions,
        hidden_states=res.hidden_states,
        last_hidden_state=res.last_hidden_state,
    )


def question_answering_adaptation(res: QuestionAnsweringModelOutput) -> QuestionAnsweringOutput:
    r""" Convert transformers QuestionAnsweringModelOutput to our QuestionAnsweringOutput. """
    if isinstance(res, QuestionAnsweringOutput):
        return res

    return QuestionAnsweringOutput(
        attentions=res.attentions,
        hidden_states=res.hidden_states,
        question_answering_loss=res.loss,
        start_position_logits=res.start_logits,
        end_position_logits=res.end_logits,
    )


def masked_lm_adaptation(res: TransformersMaskedLMOutput) -> MaskedLMOutput:
    r""" Convert transformers HF MaskedLMOutput to our MaskedLMOutput. """
    if isinstance(res, MaskedLMOutput):
        return res

    return MaskedLMOutput(
        last_hidden_state=None,
        hidden_states=res.hidden_states,
        attentions=res.attentions,
        masked_lm_loss=res.loss,
        masked_lm_logits=res.logits,
    )


def token_detection_adaptation(res: ElectraForPreTrainingOutput) -> TokenDetectionOutput:
    r""" Convert transformers ElectraForPreTrainingOutput to our TokenDetectionOutput. """
    if isinstance(res, TokenDetectionOutput):
        return res

    return TokenDetectionOutput(
        last_hidden_state=None,
        hidden_states=res.hidden_states,
        attentions=res.attentions,
        token_detection_loss=res.loss,
        token_detection_logits=res.logits,
    )


def token_classification_adaptation(res: TokenClassifierOutput) -> TokenClassOutput:
    r""" Convert transformers TokenClassifierOutput to our TokenClassificationOutput. """
    if isinstance(res, TokenClassOutput):
        return res

    return TokenClassOutput(
        last_hidden_state=None,
        hidden_states=res.hidden_states,
        attentions=res.attentions,
        token_class_loss=res.loss,
        token_class_logits=res.logits,
    )


def seq_to_seq_lm_adaptation(res: TransformersSeq2SeqLMOutput) -> SeqToSeqLMOutput:
    r""" Convert transformers HF SeqToSeqLMOutput to our SeqToSeqLMOutput. """
    if isinstance(res, SeqToSeqLMOutput):
        return res

    return SeqToSeqLMOutput(
        seq_to_seq_lm_loss=res.loss,
        seq_to_seq_lm_logits=res.logits,
        last_hidden_state=res.encoder_last_hidden_state,
        hidden_states=res.encoder_hidden_states,
        attentions=res.encoder_attentions,
        past_key_values=res.past_key_values,
        decoder_hidden_states=res.decoder_hidden_states,
        decoder_attentions=res.decoder_attentions,
        cross_attentions=res.cross_attentions,
    )


def masked_lm_and_token_detection_adaptation(
    res: MaskedLMAndTokenClassOutput
) -> MaskedLMAndTokenClassOutput:
    r""" Masked LM and Token Adaptation exists only in this framework, so no conversion is needed. """
    return res


def adapt_label_names_to_transformers(kwargs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    r""" Adapt label names for transformer models. """

    labels_keys = [k for k in kwargs.keys() if 'labels' in k]

    assert all(k in ADMISSIBLE_LABEL_NAMES for k in labels_keys), (  # nosec
        f"encountered keys not among allowed. got {labels_keys}, expected keys in {ADMISSIBLE_LABEL_NAMES}"
    )

    # working with original transformer models, just a single 'labels' keyword for every task
    if len(labels_keys) == 1:
        kwargs['labels'] = kwargs.pop(labels_keys[0])
    if 'start_position_labels' in kwargs:
        kwargs['start_positions'] = kwargs.pop('start_position_labels')
    if 'end_position_labels' in kwargs:
        kwargs['end_positions'] = kwargs.pop('end_position_labels')

    # otherwise we are working with multiple classification models, which accept more than a label argument
    # using specific names

    return kwargs

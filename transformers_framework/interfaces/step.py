from dataclasses import dataclass
from typing import List

import torch


@dataclass
class StepOutput:
    loss: torch.Tensor


@dataclass
class MaskedLMStepOutput(StepOutput):
    masked_lm_loss: torch.Tensor
    masked_lm_predictions: torch.Tensor
    masked_lm_logits: torch.Tensor
    masked_lm_labels: torch.Tensor


@dataclass
class SeqToSeqMaskedLMStepOutput(StepOutput):
    seq_to_seq_lm_loss: torch.Tensor
    seq_to_seq_lm_predictions: torch.Tensor
    seq_to_seq_lm_labels: torch.Tensor


@dataclass
class SeqToSeqGenStepOutput(StepOutput):
    generation_input_ids: torch.Tensor
    generation_labels: List


@dataclass
class SeqClassStepOutput(StepOutput):
    seq_class_loss: torch.Tensor
    seq_class_predictions: torch.Tensor
    seq_class_labels: torch.Tensor


@dataclass
class AnswerSelectionStepOutput(SeqClassStepOutput):
    seq_class_index: torch.Tensor
    seq_class_scores: torch.Tensor


@dataclass
class TokenClassStepOutput(StepOutput):
    token_class_loss: torch.Tensor
    token_class_predictions: torch.Tensor
    token_class_logits: torch.Tensor
    token_class_labels: torch.Tensor


@dataclass
class QuestionAnsweringStepOutput(StepOutput):
    question_answering_index: torch.Tensor
    question_answering_loss: torch.Tensor
    question_answering_start_logits: torch.Tensor
    question_answering_end_logits: torch.Tensor
    question_answering_start_predictions: torch.Tensor
    question_answering_end_predictions: torch.Tensor
    question_answering_start_labels: torch.Tensor
    question_answering_end_labels: torch.Tensor
    question_answering_tokens: List
    question_answering_covered_tokens: List
    question_answering_token_is_max_context: List
    question_answering_offset_mapping: List
    question_answering_context: List
    question_answering_gold_answers: List


@dataclass
class TokenDetectionStepOutput(StepOutput):
    token_detection_loss: torch.Tensor
    token_detection_predictions: torch.Tensor
    token_detection_logits: torch.Tensor
    token_detection_labels: torch.Tensor


@dataclass
class TokenDetectionAndSeqClassStepOutput(TokenDetectionStepOutput, SeqClassStepOutput):
    ...


@dataclass
class MaskedLMAndAnswerSelectionStepOutput(
    AnswerSelectionStepOutput, MaskedLMStepOutput
):
    ...


@dataclass
class MaskedLMAndSeqClassStepOutput(SeqClassStepOutput, MaskedLMStepOutput):
    ...


@dataclass
class MaskedLMAndTokenClassStepOutput(
    TokenClassStepOutput, MaskedLMStepOutput
):
    ...


@dataclass
class MaskedLMAndQuestionAnsweringStepOutput(QuestionAnsweringStepOutput, MaskedLMStepOutput):
    ...


@dataclass
class MaskedLMAndTokenDetectionStepOutput(MaskedLMStepOutput, TokenDetectionStepOutput):
    ...


@dataclass
class MaskedLMAndTokenDetectionAndSeqClassStepOutput(
    MaskedLMStepOutput, SeqClassStepOutput, TokenDetectionStepOutput
):
    ...

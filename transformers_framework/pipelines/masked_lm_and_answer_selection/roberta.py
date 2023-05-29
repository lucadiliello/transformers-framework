from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers_framework.architectures.roberta.configuration_roberta import RobertaExtendedConfig
from transformers_framework.architectures.roberta.modeling_roberta import (
    RobertaForMaskedLMAndExtendedSequenceClassification,
    RobertaForMaskedLMAndSequenceClassification,
)
from transformers_framework.architectures.roberta.tokenization_roberta import RobertaExtendedTokenizerFast
from transformers_framework.pipelines.masked_lm_and_answer_selection.base import MaskedLMAndAnswerSelectionPipeline


class RobertaMaskedLMAndAnswerSelectionPipeline(
    MaskedLMAndAnswerSelectionPipeline
):

    CONFIG_CLASS = RobertaConfig
    CONFIG_EXTENDED_CLASS = RobertaExtendedConfig
    MODEL_CLASS = RobertaForMaskedLMAndSequenceClassification
    MODEL_EXTENDED_CLASS = RobertaForMaskedLMAndExtendedSequenceClassification
    TOKENIZER_CLASS = RobertaTokenizerFast
    TOKENIZER_EXTENDED_CLASS = RobertaExtendedTokenizerFast

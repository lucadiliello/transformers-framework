from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers_framework.architectures.roberta.configuration_roberta import RobertaExtendedConfig
from transformers_framework.architectures.roberta.modeling_extended_roberta import (
    RobertaForExtendedSequenceClassification,
)
from transformers_framework.architectures.roberta.tokenization_roberta import RobertaExtendedTokenizerFast
from transformers_framework.pipelines.answer_selection.base import AnswerSelectionPipeline


class RobertaAnswerSelectionPipeline(AnswerSelectionPipeline):

    CONFIG_CLASS = RobertaConfig
    CONFIG_EXTENDED_CLASS = RobertaExtendedConfig
    MODEL_CLASS = RobertaForSequenceClassification
    MODEL_EXTENDED_CLASS = RobertaForExtendedSequenceClassification
    TOKENIZER_CLASS = RobertaTokenizerFast
    TOKENIZER_EXTENDED_CLASS = RobertaExtendedTokenizerFast

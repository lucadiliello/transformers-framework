from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers_framework.architectures.roberta.configuration_roberta import RobertaExtendedConfig
from transformers_framework.architectures.roberta.modeling_extended_roberta import (
    RobertaForMaskedLMAndExtendedSequenceClassification,
)
from transformers_framework.architectures.roberta.modeling_roberta import RobertaForMaskedLMAndSequenceClassification
from transformers_framework.architectures.roberta.tokenization_roberta import RobertaExtendedTokenizerFast
from transformers_framework.pipelines.masked_lm_and_seq_class.base import MaskedLMAndSeqClassPipeline


class RobertaMaskedLMAndSeqClassPipeline(MaskedLMAndSeqClassPipeline):

    CONFIG_CLASS = RobertaConfig
    CONFIG_EXTENDED_CLASS = RobertaExtendedConfig
    MODEL_CLASS = RobertaForMaskedLMAndSequenceClassification
    MODEL_EXTENDED_CLASS = RobertaForMaskedLMAndExtendedSequenceClassification
    TOKENIZER_CLASS = RobertaTokenizerFast
    TOKENIZER_EXTENDED_CLASS = RobertaExtendedTokenizerFast

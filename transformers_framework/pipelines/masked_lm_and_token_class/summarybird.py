from transformers_framework.architectures.summarybird.configuration_summarybird import SummaryBirdConfig
from transformers_framework.architectures.summarybird.modeling_summarybird import (
    SummaryBirdForMaskedLMAndTokenClassification,
)
from transformers_framework.architectures.summarybird.tokenization_summarybird import SummaryBirdTokenizerFast
from transformers_framework.pipelines.masked_lm_and_token_class.base import MaskedLMAndTokenClass


class SummaryBirdMaskedLMAndTokenClass(MaskedLMAndTokenClass):

    CONFIG_CLASS = SummaryBirdConfig
    MODEL_CLASS = SummaryBirdForMaskedLMAndTokenClassification
    TOKENIZER_CLASS = SummaryBirdTokenizerFast

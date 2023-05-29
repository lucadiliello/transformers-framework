from transformers_framework.architectures.summarybird.configuration_summarybird import SummaryBirdConfig
from transformers_framework.architectures.summarybird.modeling_summarybird import SummaryBirdForMaskedLM
from transformers_framework.architectures.summarybird.tokenization_summarybird import SummaryBirdTokenizerFast
from transformers_framework.pipelines.masked_lm.base import MaskedLMPipeline


class SummaryBirdMaskedLMPipeline(MaskedLMPipeline):

    CONFIG_CLASS = SummaryBirdConfig
    MODEL_CLASS = SummaryBirdForMaskedLM
    TOKENIZER_CLASS = SummaryBirdTokenizerFast

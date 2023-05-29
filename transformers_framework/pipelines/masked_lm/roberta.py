from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers_framework.pipelines.masked_lm.base import MaskedLMPipeline


class RobertaMaskedLMPipeline(MaskedLMPipeline):

    CONFIG_CLASS = RobertaConfig
    MODEL_CLASS = MODEL_EXTENDED_CLASS = RobertaForMaskedLM
    TOKENIZER_CLASS = RobertaTokenizerFast

from transformers.models.deberta.configuration_deberta import DebertaConfig
from transformers.models.deberta.modeling_deberta import DebertaForMaskedLM
from transformers.models.deberta.tokenization_deberta_fast import DebertaTokenizerFast

from transformers_framework.pipelines.masked_lm.base import MaskedLMPipeline


class DebertaMaskedLMPipeline(MaskedLMPipeline):

    CONFIG_CLASS = DebertaConfig
    MODEL_CLASS = DebertaForMaskedLM
    TOKENIZER_CLASS = DebertaTokenizerFast

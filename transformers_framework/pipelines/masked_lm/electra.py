from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.models.electra.modeling_electra import ElectraForMaskedLM
from transformers.models.electra.tokenization_electra_fast import ElectraTokenizerFast

from transformers_framework.pipelines.masked_lm.base import MaskedLMPipeline


class ElectraMaskedLMPipeline(MaskedLMPipeline):

    CONFIG_CLASS = ElectraConfig
    MODEL_CLASS = ElectraForMaskedLM
    TOKENIZER_CLASS = ElectraTokenizerFast

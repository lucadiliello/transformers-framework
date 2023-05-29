from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.models.electra.modeling_electra import ElectraForTokenClassification
from transformers.models.electra.tokenization_electra_fast import ElectraTokenizerFast

from transformers_framework.pipelines.token_class.base import TokenClassPipeline


class ElectraTokenClassPipeline(TokenClassPipeline):
    r""" ELECTRA with a simple TC binary classification head on top. """

    CONFIG_CLASS = ElectraConfig
    MODEL_CLASS = ElectraForTokenClassification
    TOKENIZER_CLASS = ElectraTokenizerFast

from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.models.electra.modeling_electra import ElectraForPreTraining
from transformers.models.electra.tokenization_electra_fast import ElectraTokenizerFast

from transformers_framework.pipelines.random_token_detection.base import RandomTokenDetectionPipeline


class ElectraRandomTokenDetectionPipeline(RandomTokenDetectionPipeline):
    r""" ELECTRA with a simple RTD binary classification head on top. """

    CONFIG_CLASS = ElectraConfig
    MODEL_CLASS = ElectraForPreTraining
    TOKENIZER_CLASS = ElectraTokenizerFast

from transformers.models.roberta import RobertaConfig

from transformers_framework.architectures.roberta.modeling_roberta import RobertaForTokenDetection
from transformers_framework.architectures.roberta.tokenization_roberta import RobertaExtendedTokenizerFast
from transformers_framework.pipelines.cluster_random_token_detection.base import ClusterRandomTokenDetectionPipeline


class RobertaClusterRandomTokenDetectionPipeline(ClusterRandomTokenDetectionPipeline):
    r""" Roberta with a simple RTD binary classification head on top. """

    CONFIG_CLASS = RobertaConfig
    MODEL_CLASS = RobertaForTokenDetection
    TOKENIZER_CLASS = RobertaExtendedTokenizerFast

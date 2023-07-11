from transformers.models.bert.configuration_bert import BertConfig

from transformers_framework.architectures.bert.modeling_bert import BertForTokenDetection
from transformers_framework.architectures.bert.tokenization_bert import BertExtendedTokenizerFast
from transformers_framework.pipelines.cluster_random_token_detection.base import ClusterRandomTokenDetectionPipeline


class BertClusterRandomTokenDetectionPipeline(ClusterRandomTokenDetectionPipeline):
    r""" Bert with a simple C-RTD binary classification head on top. """

    CONFIG_CLASS = BertConfig
    MODEL_CLASS = BertForTokenDetection
    TOKENIZER_CLASS = BertExtendedTokenizerFast

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForTokenClassification
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from transformers_framework.pipelines.token_class.base import TokenClassPipeline


class BertTokenClassPipeline(TokenClassPipeline):
    r""" Bert with a simple TC binary classification head on top. """

    CONFIG_CLASS = BertConfig
    MODEL_CLASS = BertForTokenClassification
    TOKENIZER_CLASS = BertTokenizerFast

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification

from transformers_framework.architectures.bert.tokenization_bert import BertExtendedTokenizerFast
from transformers_framework.pipelines.seq_class.base import SeqClassPipeline


class BertSeqClassPipeline(SeqClassPipeline):

    CONFIG_CLASS = BertConfig
    MODEL_CLASS = BertForSequenceClassification
    TOKENIZER_CLASS = BertExtendedTokenizerFast

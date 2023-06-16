from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from transformers_framework.architectures.bert.modeling_bert import BertForMaskedLMAndSequenceClassification
from transformers_framework.architectures.bert.tokenization_bert import BertExtendedTokenizerFast
from transformers_framework.pipelines.masked_lm_and_seq_class.base import MaskedLMAndSeqClassPipeline


class BertMaskedLMAndSeqClassPipeline(MaskedLMAndSeqClassPipeline):

    CONFIG_CLASS = BertConfig
    MODEL_CLASS = BertForMaskedLMAndSequenceClassification
    TOKENIZER_CLASS = BertTokenizerFast
    TOKENIZER_EXTENDED_CLASS = BertExtendedTokenizerFast

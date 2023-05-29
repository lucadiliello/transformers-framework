from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from transformers_framework.pipelines.masked_lm.base import MaskedLMPipeline


class BertMaskedLMPipeline(MaskedLMPipeline):

    CONFIG_CLASS = BertConfig
    MODEL_CLASS = BertForMaskedLM
    TOKENIZER_CLASS = BertTokenizerFast

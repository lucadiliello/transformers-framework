from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from transformers_framework.architectures.bert.tokenization_bert import BertExtendedTokenizerFast
from transformers_framework.pipelines.answer_selection.base import AnswerSelectionPipeline


class BertAnswerSelectionPipeline(AnswerSelectionPipeline):

    CONFIG_CLASS = BertConfig
    MODEL_CLASS = BertForSequenceClassification
    TOKENIZER_CLASS = BertTokenizerFast
    TOKENIZER_EXTENDED_CLASS = BertExtendedTokenizerFast

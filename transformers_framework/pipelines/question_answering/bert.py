from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForQuestionAnswering
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from transformers_framework.pipelines.question_answering.base import QuestionAnsweringPipeline


class BertQuestionAnsweringPipeline(QuestionAnsweringPipeline):

    CONFIG_CLASS = BertConfig
    MODEL_CLASS = BertForQuestionAnswering
    TOKENIZER_CLASS = BertTokenizerFast

from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaForQuestionAnswering
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers_framework.pipelines.question_answering.base import QuestionAnsweringPipeline


class RobertaQuestionAnsweringPipeline(QuestionAnsweringPipeline):

    CONFIG_CLASS = RobertaConfig
    MODEL_CLASS = RobertaForQuestionAnswering
    TOKENIZER_CLASS = RobertaTokenizerFast

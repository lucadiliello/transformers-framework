from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2ForQuestionAnswering
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast

from transformers_framework.pipelines.question_answering.base import QuestionAnsweringPipeline


class DebertaV2QuestionAnsweringPipeline(QuestionAnsweringPipeline):

    CONFIG_CLASS = DebertaV2Config
    MODEL_CLASS = DebertaV2ForQuestionAnswering
    TOKENIZER_CLASS = DebertaV2TokenizerFast

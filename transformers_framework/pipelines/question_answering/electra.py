from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.models.electra.modeling_electra import ElectraForQuestionAnswering
from transformers.models.electra.tokenization_electra_fast import ElectraTokenizerFast

from transformers_framework.pipelines.question_answering.base import QuestionAnsweringPipeline


class ElectraQuestionAnsweringPipeline(QuestionAnsweringPipeline):

    CONFIG_CLASS = ElectraConfig
    MODEL_CLASS = ElectraForQuestionAnswering
    TOKENIZER_CLASS = ElectraTokenizerFast

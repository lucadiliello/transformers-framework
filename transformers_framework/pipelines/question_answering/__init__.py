from transformers_framework.pipelines.question_answering.bert import BertQuestionAnsweringPipeline
from transformers_framework.pipelines.question_answering.deberta_v2 import DebertaV2QuestionAnsweringPipeline
from transformers_framework.pipelines.question_answering.electra import ElectraQuestionAnsweringPipeline
from transformers_framework.pipelines.question_answering.roberta import RobertaQuestionAnsweringPipeline


models = dict(
    roberta=RobertaQuestionAnsweringPipeline,
    bert=BertQuestionAnsweringPipeline,
    electra=ElectraQuestionAnsweringPipeline,
    deberta_v2=DebertaV2QuestionAnsweringPipeline,
)

from transformers_framework.pipelines.answer_selection.bert import BertAnswerSelectionPipeline
from transformers_framework.pipelines.answer_selection.deberta import DebertaAnswerSelectionPipeline
from transformers_framework.pipelines.answer_selection.deberta_v2 import DebertaV2AnswerSelectionPipeline
from transformers_framework.pipelines.answer_selection.electra import ElectraAnswerSelectionPipeline
from transformers_framework.pipelines.answer_selection.roberta import (
    RobertaAnswerSelectionPipeline,
    WeightedRobertaAnswerSelectionPipeline,
)
from transformers_framework.pipelines.answer_selection.xlm import XLMRobertaAnswerSelectionPipeline


models = dict(
    bert=BertAnswerSelectionPipeline,
    roberta=RobertaAnswerSelectionPipeline,
    weighted_roberta=WeightedRobertaAnswerSelectionPipeline,
    deberta=DebertaAnswerSelectionPipeline,
    deberta_v2=DebertaV2AnswerSelectionPipeline,
    electra=ElectraAnswerSelectionPipeline,
    xlm_roberta=XLMRobertaAnswerSelectionPipeline,
)

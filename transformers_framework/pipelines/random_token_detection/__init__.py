from transformers_framework.pipelines.random_token_detection.bert import BertRandomTokenDetectionPipeline
from transformers_framework.pipelines.random_token_detection.electra import ElectraRandomTokenDetectionPipeline
from transformers_framework.pipelines.random_token_detection.roberta import RobertaRandomTokenDetectionPipeline


models = dict(
    bert=BertRandomTokenDetectionPipeline,
    roberta=RobertaRandomTokenDetectionPipeline,
    electra=ElectraRandomTokenDetectionPipeline,
)

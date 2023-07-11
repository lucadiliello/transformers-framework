from transformers_framework.pipelines.cluster_random_token_detection.bert import BertClusterRandomTokenDetectionPipeline
from transformers_framework.pipelines.cluster_random_token_detection.roberta import (
    RobertaClusterRandomTokenDetectionPipeline,
)


models = dict(
    bert=BertClusterRandomTokenDetectionPipeline,
    roberta=RobertaClusterRandomTokenDetectionPipeline,
)

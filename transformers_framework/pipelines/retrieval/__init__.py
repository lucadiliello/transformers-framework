from transformers_framework.pipelines.retrieval.bert import BertRetrievalPipeline
from transformers_framework.pipelines.retrieval.deberta import DebertaRetrievalPipeline
from transformers_framework.pipelines.retrieval.deberta_v2 import DebertaV2RetrievalPipeline
from transformers_framework.pipelines.retrieval.roberta import RobertaRetrievalPipeline


models = dict(
    bert=BertRetrievalPipeline,
    roberta=RobertaRetrievalPipeline,
    deberta=DebertaRetrievalPipeline,
    deberta_v2=DebertaV2RetrievalPipeline,
)

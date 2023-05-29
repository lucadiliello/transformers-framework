from transformers_framework.pipelines.token_class.bert import BertTokenClassPipeline
from transformers_framework.pipelines.token_class.electra import ElectraTokenClassPipeline
from transformers_framework.pipelines.token_class.roberta import RobertaTokenClassPipeline


models = dict(
    bert=BertTokenClassPipeline,
    roberta=RobertaTokenClassPipeline,
    electra=ElectraTokenClassPipeline,
)

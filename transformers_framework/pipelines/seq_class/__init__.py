from transformers_framework.pipelines.seq_class.albert import AlbertSeqClassPipeline
from transformers_framework.pipelines.seq_class.bart import BartSeqClassPipeline
from transformers_framework.pipelines.seq_class.bert import BertSeqClassPipeline
from transformers_framework.pipelines.seq_class.deberta import DebertaSeqClassPipeline
from transformers_framework.pipelines.seq_class.electra import ElectraSeqClassPipeline
from transformers_framework.pipelines.seq_class.roberta import RobertaSeqClassPipeline


models = dict(
    albert=AlbertSeqClassPipeline,
    bert=BertSeqClassPipeline,
    roberta=RobertaSeqClassPipeline,
    electra=ElectraSeqClassPipeline,
    bart=BartSeqClassPipeline,
    deberta=DebertaSeqClassPipeline,
)

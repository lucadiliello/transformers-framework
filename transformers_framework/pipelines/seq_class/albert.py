from transformers.models.albert.configuration_albert import AlbertConfig
from transformers.models.albert.modeling_albert import AlbertForSequenceClassification
from transformers.models.albert.tokenization_albert_fast import AlbertTokenizerFast

from transformers_framework.pipelines.seq_class.base import SeqClassPipeline


class AlbertSeqClassPipeline(SeqClassPipeline):

    CONFIG_CLASS = AlbertConfig
    MODEL_CLASS = AlbertForSequenceClassification
    TOKENIZER_CLASS = AlbertTokenizerFast

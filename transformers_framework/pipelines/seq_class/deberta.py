from transformers.models.deberta.configuration_deberta import DebertaConfig
from transformers.models.deberta.modeling_deberta import DebertaForSequenceClassification
from transformers.models.deberta.tokenization_deberta_fast import DebertaTokenizerFast

from transformers_framework.pipelines.seq_class.base import SeqClassPipeline


class DebertaSeqClassPipeline(SeqClassPipeline):

    CONFIG_CLASS = DebertaConfig
    MODEL_CLASS = DebertaForSequenceClassification
    TOKENIZER_CLASS = DebertaTokenizerFast

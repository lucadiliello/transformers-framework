from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartForSequenceClassification
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast

from transformers_framework.pipelines.seq_class.base import SeqClassPipeline


class BartSeqClassPipeline(SeqClassPipeline):

    CONFIG_CLASS = BartConfig
    MODEL_CLASS = BartForSequenceClassification
    TOKENIZER_CLASS = BartTokenizerFast

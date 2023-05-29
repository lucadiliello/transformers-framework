from transformers.models.deberta.configuration_deberta import DebertaConfig
from transformers.models.deberta.tokenization_deberta_fast import DebertaTokenizerFast

from transformers_framework.architectures.deberta.modeling_deberta import DebertaForMaskedLMAndSequenceClassification
from transformers_framework.pipelines.masked_lm_and_seq_class.base import MaskedLMAndSeqClassPipeline


class DebertaMaskedLMAndSeqClassPipeline(MaskedLMAndSeqClassPipeline):

    CONFIG_CLASS = DebertaConfig
    MODEL_CLASS = DebertaForMaskedLMAndSequenceClassification
    TOKENIZER_CLASS = DebertaTokenizerFast

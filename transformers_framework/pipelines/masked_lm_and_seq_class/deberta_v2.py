from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast

from transformers_framework.architectures.deberta_v2.modeling_deberta_v2 import (
    DebertaV2ForMaskedLMAndSequenceClassification,
)
from transformers_framework.pipelines.masked_lm_and_seq_class.base import MaskedLMAndSeqClassPipeline


class DebertaV2MaskedLMAndSeqClassPipeline(MaskedLMAndSeqClassPipeline):

    CONFIG_CLASS = DebertaV2Config
    MODEL_CLASS = DebertaV2ForMaskedLMAndSequenceClassification
    TOKENIZER_CLASS = DebertaV2TokenizerFast

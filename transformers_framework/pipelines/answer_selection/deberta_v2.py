from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2ForSequenceClassification
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast

from transformers_framework.architectures.deberta_v2.configuration_deberta_v2 import DebertaV2ExtendedConfig
from transformers_framework.architectures.deberta_v2.modeling_extended_deberta_v2 import (
    DebertaV2ForExtendedSequenceClassification,
)
from transformers_framework.architectures.deberta_v2.tokenization_deberta_v2 import DebertaV2ExtendedTokenizerFast
from transformers_framework.pipelines.answer_selection.base import AnswerSelectionPipeline


class DebertaV2AnswerSelectionPipeline(AnswerSelectionPipeline):

    CONFIG_CLASS = DebertaV2Config
    CONFIG_EXTENDED_CLASS = DebertaV2ExtendedConfig
    MODEL_CLASS = DebertaV2ForSequenceClassification
    MODEL_EXTENDED_CLASS = DebertaV2ForExtendedSequenceClassification
    TOKENIZER_CLASS = DebertaV2TokenizerFast
    TOKENIZER_EXTENDED_CLASS = DebertaV2ExtendedTokenizerFast

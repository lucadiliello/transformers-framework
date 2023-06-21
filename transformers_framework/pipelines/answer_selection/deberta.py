from transformers.models.deberta.configuration_deberta import DebertaConfig
from transformers.models.deberta.modeling_deberta import DebertaForSequenceClassification
from transformers.models.deberta.tokenization_deberta_fast import DebertaTokenizerFast
from transformers_framework.architectures.deberta.tokenization_deberta import DebertaExtendedTokenizerFast

from transformers_framework.pipelines.answer_selection.base import AnswerSelectionPipeline


class DebertaAnswerSelectionPipeline(AnswerSelectionPipeline):

    CONFIG_CLASS = DebertaConfig
    MODEL_CLASS = DebertaForSequenceClassification
    TOKENIZER_CLASS = DebertaTokenizerFast
    TOKENIZER_EXTENDED_CLASS = DebertaExtendedTokenizerFast

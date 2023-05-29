from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.models.electra.modeling_electra import ElectraForSequenceClassification
from transformers.models.electra.tokenization_electra_fast import ElectraTokenizerFast

from transformers_framework.architectures.electra.tokenization_electra import ElectraExtendedTokenizerFast
from transformers_framework.pipelines.answer_selection.base import AnswerSelectionPipeline


class ElectraAnswerSelectionPipeline(AnswerSelectionPipeline):

    CONFIG_CLASS = ElectraConfig
    MODEL_CLASS = ElectraForSequenceClassification
    TOKENIZER_CLASS = ElectraTokenizerFast
    TOKENIZER_EXTENDED_CLASS = ElectraExtendedTokenizerFast

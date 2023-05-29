from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers_framework.architectures.roberta.modeling_roberta import RobertaForMaskedLMAndTokenClassification
from transformers_framework.pipelines.masked_lm_and_token_class.base import MaskedLMAndTokenClass


class RobertaMaskedLMAndTokenClass(MaskedLMAndTokenClass):

    CONFIG_CLASS = RobertaConfig
    MODEL_CLASS = RobertaForMaskedLMAndTokenClassification
    TOKENIZER_CLASS = RobertaTokenizerFast

from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaForTokenClassification
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers_framework.pipelines.token_class.base import TokenClassPipeline


class RobertaTokenClassPipeline(TokenClassPipeline):
    r""" RoBERTa with a simple TC binary classification head on top. """

    CONFIG_CLASS = RobertaConfig
    MODEL_CLASS = RobertaForTokenClassification
    TOKENIZER_CLASS = RobertaTokenizerFast

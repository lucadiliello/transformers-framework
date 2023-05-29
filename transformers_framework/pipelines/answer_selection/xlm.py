from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaForSequenceClassification
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast

from transformers_framework.pipelines.answer_selection.base import AnswerSelectionPipeline


class XLMRobertaAnswerSelectionPipeline(AnswerSelectionPipeline):

    CONFIG_CLASS = XLMRobertaConfig
    MODEL_CLASS = XLMRobertaForSequenceClassification
    TOKENIZER_CLASS = XLMRobertaTokenizerFast

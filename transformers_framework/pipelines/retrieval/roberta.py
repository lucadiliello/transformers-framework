from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers_framework.pipelines.retrieval.base import RetrievalPipeline


class RobertaRetrievalPipeline(RetrievalPipeline):

    CONFIG_CLASS = RobertaConfig
    MODEL_CLASS = RobertaModel
    TOKENIZER_CLASS = RobertaTokenizerFast

from transformers.models.deberta.configuration_deberta import DebertaConfig
from transformers.models.deberta.modeling_deberta import DebertaModel
from transformers.models.deberta.tokenization_deberta_fast import DebertaTokenizerFast

from transformers_framework.pipelines.retrieval.base import RetrievalPipeline


class DebertaRetrievalPipeline(RetrievalPipeline):

    CONFIG_CLASS = DebertaConfig
    MODEL_CLASS = DebertaModel
    TOKENIZER_CLASS = DebertaTokenizerFast

from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast

from transformers_framework.pipelines.retrieval.base import RetrievalPipeline


class DebertaV2RetrievalPipeline(RetrievalPipeline):

    CONFIG_CLASS = DebertaV2Config
    MODEL_CLASS = DebertaV2Model
    TOKENIZER_CLASS = DebertaV2TokenizerFast

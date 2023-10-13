from sentence_transformers import SentenceTransformer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from transformers_framework.pipelines.retrieval.base import RetrievalPipeline


class BertRetrievalPipeline(RetrievalPipeline):

    CONFIG_CLASS = BertConfig
    MODEL_CLASS = BertModel
    TOKENIZER_CLASS = BertTokenizerFast

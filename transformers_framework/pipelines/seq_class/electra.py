from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.models.electra.modeling_electra import ElectraForSequenceClassification

from transformers_framework.architectures.electra.tokenization_electra import ElectraExtendedTokenizerFast
from transformers_framework.pipelines.seq_class.base import SeqClassPipeline


class ElectraSeqClassPipeline(SeqClassPipeline):

    CONFIG_CLASS = ElectraConfig
    MODEL_CLASS = ElectraForSequenceClassification
    TOKENIZER_CLASS = ElectraExtendedTokenizerFast

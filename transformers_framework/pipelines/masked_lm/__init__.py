from transformers_framework.pipelines.masked_lm.bert import BertMaskedLMPipeline
from transformers_framework.pipelines.masked_lm.deberta import DebertaMaskedLMPipeline
from transformers_framework.pipelines.masked_lm.electra import ElectraMaskedLMPipeline
from transformers_framework.pipelines.masked_lm.roberta import RobertaMaskedLMPipeline
from transformers_framework.pipelines.masked_lm.summarybird import SummaryBirdMaskedLMPipeline


models = dict(
    bert=BertMaskedLMPipeline,
    roberta=RobertaMaskedLMPipeline,
    deberta=DebertaMaskedLMPipeline,
    electra=ElectraMaskedLMPipeline,
    summarybird=SummaryBirdMaskedLMPipeline,
)

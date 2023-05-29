from transformers_framework.pipelines.masked_lm_and_token_class.roberta import RobertaMaskedLMAndTokenClass
from transformers_framework.pipelines.masked_lm_and_token_class.summarybird import SummaryBirdMaskedLMAndTokenClass


models = dict(
    roberta=RobertaMaskedLMAndTokenClass,
    summarybird=SummaryBirdMaskedLMAndTokenClass,
)

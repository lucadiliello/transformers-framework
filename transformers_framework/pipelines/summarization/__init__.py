from transformers_framework.pipelines.summarization.bart import (
    BartMultiTokenSummarizationPipeline,
    BartSummarizationPipeline,
)
from transformers_framework.pipelines.summarization.t5 import T5MultiTokenSummarizationPipeline, T5SummarizationPipeline


models = dict(
    t5=T5SummarizationPipeline,
    bart=BartSummarizationPipeline,
    multi_token_t5=T5MultiTokenSummarizationPipeline,
    multi_token_bart=BartMultiTokenSummarizationPipeline,
)

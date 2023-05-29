from transformers_framework.pipelines.denoising.bart import BartDenoisingPipeline, BartMultiTokenDenoisingPipeline
from transformers_framework.pipelines.denoising.t5 import T5DenoisingPipeline, T5MultiTokenDenoisingPipeline


models = dict(
    t5=T5DenoisingPipeline,
    multi_token_t5=T5MultiTokenDenoisingPipeline,
    bart=BartDenoisingPipeline,
    multi_token_bart=BartMultiTokenDenoisingPipeline,
)

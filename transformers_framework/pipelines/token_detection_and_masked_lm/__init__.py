from transformers_framework.pipelines.token_detection_and_masked_lm.deberta_v2 import (
    DebertaV2TokenDetectionAndMaskedLMPipeline,
)
from transformers_framework.pipelines.token_detection_and_masked_lm.electra import (
    ElectraTokenDetectionAndMaskedLMPipeline,
)


models = dict(
    deberta_v2=DebertaV2TokenDetectionAndMaskedLMPipeline,
    electra=ElectraTokenDetectionAndMaskedLMPipeline,
)

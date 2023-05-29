from transformers_framework.pipelines.token_detection_and_masked_lm_and_seq_class.deberta_v2 import \
    DebertaV2TokenDetectionAndMaskedLMAndSeqClassPipeline  # noqa: E501
from transformers_framework.pipelines.token_detection_and_masked_lm_and_seq_class.electra import (
    ElectraTokenDetectionAndMaskedLMAndSeqClassPipeline,
)


models = dict(
    deberta_v2=DebertaV2TokenDetectionAndMaskedLMAndSeqClassPipeline,
    electra=ElectraTokenDetectionAndMaskedLMAndSeqClassPipeline,
)

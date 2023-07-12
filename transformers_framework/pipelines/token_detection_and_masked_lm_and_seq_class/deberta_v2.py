from transformers.configuration_utils import PretrainedConfig
from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2ForMaskedLM
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
from transformers_framework.architectures.deberta_v2.configuration_deberta_v2 import DebertaV2ExtendedConfig

from transformers_framework.architectures.deberta_v2.modeling_deberta_v2 import (
    DebertaV2ForPreTrainingAndSequenceClassification,
)
from transformers_framework.architectures.deberta_v2.modeling_extended_deberta_v2 import (
    DebertaV2ForPreTrainingAndExtendedSequenceClassification,
)
from transformers_framework.architectures.deberta_v2.tokenization_deberta_v2 import DebertaV2ExtendedTokenizerFast
from transformers_framework.pipelines.token_detection_and_masked_lm_and_seq_class.base import (
    TokenDetectionAndMaskedLMAndSeqClassPipeline,
)
from transformers_framework.utilities.models import get_deberta_reduced_generator_config, tie_weights_deberta


class DebertaV2TokenDetectionAndMaskedLMAndSeqClassPipeline(TokenDetectionAndMaskedLMAndSeqClassPipeline):

    CONFIG_CLASS = DebertaV2Config
    CONFIG_EXTENDED_CLASS = DebertaV2ExtendedConfig
    TOKENIZER_CLASS = DebertaV2TokenizerFast
    TOKENIZER_EXTENDED_CLASS = DebertaV2ExtendedTokenizerFast
    MODEL_CLASS = DebertaV2ForPreTrainingAndSequenceClassification
    MODEL_EXTENDED_CLASS = DebertaV2ForPreTrainingAndExtendedSequenceClassification
    GENERATOR_MODEL_CLASS = DebertaV2ForMaskedLM

    def get_generator_config(self, config: PretrainedConfig):
        return get_deberta_reduced_generator_config(
            config, factor=self.hyperparameters.generator_size
        )

    def tie_weights(self):
        tie_weights_deberta(
            self.generator,
            self.model,
            tie_generator_discriminator_embeddings=self.hyperparameters.tie_generator_discriminator_embeddings,
        )

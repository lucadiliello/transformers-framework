from typing import Dict

from transformers.configuration_utils import PretrainedConfig
from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2ForMaskedLM

from transformers_framework.architectures.deberta_v2.modeling_deberta_v2 import DebertaV2ForPreTraining
from transformers_framework.architectures.deberta_v2.tokenization_deberta_v2 import DebertaV2ExtendedTokenizerFast
from transformers_framework.pipelines.token_detection_and_masked_lm.base import TokenDetectionAndMaskedLMPipeline
from transformers_framework.utilities.models import get_deberta_reduced_generator_config, tie_weights_deberta


class DebertaV2TokenDetectionAndMaskedLMPipeline(TokenDetectionAndMaskedLMPipeline):

    CONFIG_CLASS = DebertaV2Config
    MODEL_CLASS = DebertaV2ForPreTraining
    GENERATOR_MODEL_CLASS = DebertaV2ForMaskedLM
    TOKENIZER_CLASS = DebertaV2ExtendedTokenizerFast

    def get_generator_config_parameters(self, config: PretrainedConfig, generator_size: float = 1 / 2) -> Dict:
        return get_deberta_reduced_generator_config(config, factor=generator_size)

    def tie_weights(self):
        tie_weights_deberta(
            self.generator,
            self.model,
            tie_generator_discriminator_embeddings=self.hyperparameters.tie_generator_discriminator_embeddings,
        )

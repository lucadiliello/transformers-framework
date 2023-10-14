from typing import Dict

from transformers.configuration_utils import PretrainedConfig
from transformers.models.electra.modeling_electra import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining
from transformers.models.electra.tokenization_electra_fast import ElectraTokenizerFast

from transformers_framework.pipelines.token_detection_and_masked_lm.base import TokenDetectionAndMaskedLMPipeline
from transformers_framework.utilities.models import get_electra_reduced_generator_config, tie_weights_electra


class ElectraTokenDetectionAndMaskedLMPipeline(TokenDetectionAndMaskedLMPipeline):

    CONFIG_CLASS = ElectraConfig
    MODEL_CLASS = ElectraForPreTraining
    GENERATOR_MODEL_CLASS = ElectraForMaskedLM
    TOKENIZER_CLASS = ElectraTokenizerFast

    def get_generator_config_parameters(self, config: PretrainedConfig, generator_size: float = 1 / 3) -> Dict:
        return get_electra_reduced_generator_config(config, factor=generator_size)

    def tie_weights(self):
        tie_weights_electra(
            self.generator,
            self.model,
            tie_generator_discriminator_embeddings=self.hyperparameters.tie_generator_discriminator_embeddings,
            tie_word_embeddings=self.config.tie_word_embeddings,
        )

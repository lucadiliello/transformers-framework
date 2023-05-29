from transformers.configuration_utils import PretrainedConfig
from transformers.models.electra.modeling_electra import ElectraConfig, ElectraForMaskedLM
from transformers.models.electra.tokenization_electra_fast import ElectraTokenizerFast

from transformers_framework.architectures.electra.modeling_electra import ElectraForPreTrainingAndSequenceClassification
from transformers_framework.pipelines.token_detection_and_masked_lm_and_seq_class.base import (
    TokenDetectionAndMaskedLMAndSeqClassPipeline,
)
from transformers_framework.utilities.models import get_electra_reduced_generator_config, tie_weights_electra


class ElectraTokenDetectionAndMaskedLMAndSeqClassPipeline(
    TokenDetectionAndMaskedLMAndSeqClassPipeline
):

    CONFIG_CLASS = ElectraConfig
    TOKENIZER_CLASS = ElectraTokenizerFast
    GENERATOR_MODEL_CLASS = ElectraForMaskedLM
    MODEL_CLASS = ElectraForPreTrainingAndSequenceClassification

    def get_generator_config(self, config: PretrainedConfig):
        return get_electra_reduced_generator_config(
            config, factor=self.hyperparameters.generator_size
        )

    def tie_weights(self):
        tie_weights_electra(
            self.generator,
            self.model,
            tie_generator_discriminator_embeddings=self.hyperparameters.tie_generator_discriminator_embeddings,
            tie_word_embeddings=self.config.tie_word_embeddings,
        )

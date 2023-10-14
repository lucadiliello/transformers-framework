from typing import Any, Dict, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast

from transformers_framework.architectures.t5.configuration_t5 import T5MultiTokenConfig
from transformers_framework.architectures.t5.modeling_t5 import T5ForMultiTokenConditionalGeneration
from transformers_framework.pipelines.denoising.base import DenoisingPipeline
from transformers_framework.processing.postprocessors import t5_denoising_processor
from transformers_framework.utilities.arguments import FlexibleArgumentParser, add_multi_token_arguments


class T5DenoisingPipeline(DenoisingPipeline):

    CONFIG_CLASS = T5Config
    MODEL_CLASS = T5ForConditionalGeneration
    TOKENIZER_CLASS = T5TokenizerFast

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        assert len(self.tokenizer.additional_special_tokens_ids) >= hyperparameters.max_number_of_spans, (  # nosec
            f"`max_number_of_spans` is equal to {hyperparameters.max_number_of_spans}, but this"
            f" tokenizer has only {len(self.tokenizer.additional_special_tokens_ids)} additional special tokens"
        )

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Process single samples to add denoising objective. """
        return t5_denoising_processor(
            sample=sample,
            input_column=self.hyperparameters.input_column,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            probability=self.hyperparameters.probability,
            mean_span_length=self.hyperparameters.mean_span_length,
            whole_word_denoising=self.hyperparameters.whole_word_denoising,
            max_number_of_spans=self.hyperparameters.max_number_of_spans,
        )


class T5MultiTokenDenoisingPipeline(T5DenoisingPipeline):

    CONFIG_CLASS = T5MultiTokenConfig
    MODEL_CLASS = T5ForMultiTokenConditionalGeneration

    def setup_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        r""" Just add max_output_embeddings attirbute. """
        kwargs['max_multi_token_predictions'] = self.hyperparameters.max_multi_token_predictions
        return super().setup_config(**kwargs)

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        add_multi_token_arguments(parser)
        super().add_argparse_args(parser)

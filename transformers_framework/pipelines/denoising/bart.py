from typing import Any, Dict, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast

from transformers_framework.architectures.bart.configuration_bart import BartMultiTokenConfig
from transformers_framework.architectures.bart.modeling_bart import BartForMultiTokenConditionalGeneration
from transformers_framework.pipelines.denoising.base import DenoisingPipeline
from transformers_framework.processing.postprocessors import bart_denoising_processor
from transformers_framework.utilities.arguments import FlexibleArgumentParser, add_multi_token_arguments


class BartDenoisingPipeline(DenoisingPipeline):

    CONFIG_CLASS = BartConfig
    MODEL_CLASS = BartForConditionalGeneration
    TOKENIZER_CLASS = BartTokenizerFast

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Process single samples to add denoising objective. """
        return bart_denoising_processor(
            sample=sample,
            input_column=self.hyperparameters.input_column,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            probability=self.hyperparameters.probability,
            mean_span_length=self.hyperparameters.mean_span_length,
            whole_word_denoising=self.hyperparameters.whole_word_denoising,
            max_number_of_spans=self.hyperparameters.max_number_of_spans,
            shuffle_sentences=self.hyperparameters.shuffle_sentences,
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument('--shuffle_sentences', action="store_true")


class BartMultiTokenDenoisingPipeline(BartDenoisingPipeline):

    CONFIG_CLASS = BartMultiTokenConfig
    MODEL_CLASS = BartForMultiTokenConditionalGeneration

    def setup_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        r""" Just add max_output_embeddings attirbute. """
        kwargs['max_multi_token_predictions'] = self.hyperparameters.max_multi_token_predictions
        return super().setup_config(**kwargs)

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        add_multi_token_arguments(parser)

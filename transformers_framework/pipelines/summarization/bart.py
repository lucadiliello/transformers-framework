from transformers.configuration_utils import PretrainedConfig
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast

from transformers_framework.architectures.bart.configuration_bart import BartMultiTokenConfig
from transformers_framework.architectures.bart.modeling_bart import BartForMultiTokenConditionalGeneration
from transformers_framework.pipelines.summarization.base import SummarizationPipeline
from transformers_framework.utilities.arguments import FlexibleArgumentParser, add_multi_token_arguments


class BartSummarizationPipeline(SummarizationPipeline):

    CONFIG_CLASS = BartConfig
    MODEL_CLASS = BartForConditionalGeneration
    TOKENIZER_CLASS = BartTokenizerFast


class BartMultiTokenSummarizationPipeline(BartSummarizationPipeline):

    CONFIG_CLASS = BartMultiTokenConfig
    MODEL_CLASS = BartForMultiTokenConditionalGeneration

    def configure_config(self) -> PretrainedConfig:
        r""" Just add max_output_embeddings attirbute. """
        kwargs = dict(max_multi_token_predictions=self.hyperparameters.max_multi_token_predictions)
        return super().configure_config(**kwargs)

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        add_multi_token_arguments(parser)

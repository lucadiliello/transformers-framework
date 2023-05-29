from transformers.configuration_utils import PretrainedConfig
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast

from transformers_framework.architectures.t5.configuration_t5 import T5MultiTokenConfig
from transformers_framework.architectures.t5.modeling_t5 import T5ForMultiTokenConditionalGeneration
from transformers_framework.pipelines.summarization.base import SummarizationPipeline
from transformers_framework.utilities.arguments import FlexibleArgumentParser, add_multi_token_arguments


class T5SummarizationPipeline(SummarizationPipeline):

    CONFIG_CLASS = T5Config
    MODEL_CLASS = T5ForConditionalGeneration
    TOKENIZER_CLASS = T5TokenizerFast


class T5MultiTokenSummarizationPipeline(T5SummarizationPipeline):

    CONFIG_CLASS = T5MultiTokenConfig
    MODEL_CLASS = T5ForMultiTokenConditionalGeneration

    def configure_config(self) -> PretrainedConfig:
        r""" Just add max_output_embeddings attirbute. """
        kwargs = dict(max_multi_token_predictions=self.hyperparameters.max_multi_token_predictions)
        return super().configure_config(**kwargs)

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        add_multi_token_arguments(parser)

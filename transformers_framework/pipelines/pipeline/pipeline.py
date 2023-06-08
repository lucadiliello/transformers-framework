from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union

from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers_framework.architectures.tokenization_utils import ExtendedTokenizerFast
from transformers_framework.interfaces.adaptation import adapt_label_names_to_transformers
from transformers_framework.pipelines.pipeline.mixins.metrics import MetricsMixin
from transformers_framework.pipelines.pipeline.mixins.models import ModelsMixin
from transformers_framework.pipelines.pipeline.mixins.optimizers import OptimizersMixin
from transformers_framework.pipelines.pipeline.mixins.processing import ProcessingMixin
from transformers_framework.pipelines.pipeline.mixins.properties import PropertiesMixin
from transformers_framework.utilities.arguments import FlexibleArgumentParser
from transformers_framework.utilities.datamodules import TrainerStage_to_Names
from transformers_framework.utilities.logging import parse_log_arguments
from transformers_framework.utilities.torch import clean_device_cache


DEFAULT_LOG_KWARGS = {
    RunningStage.TRAINING: dict(on_epoch=True, prog_bar=True, sync_dist=True),
    RunningStage.VALIDATING: dict(on_epoch=True, prog_bar=True, sync_dist=True),
    RunningStage.TESTING: dict(on_epoch=True, prog_bar=True, sync_dist=True),
    RunningStage.SANITY_CHECKING: dict(on_epoch=True, prog_bar=True, sync_dist=True),
}


class Pipeline(LightningModule, OptimizersMixin, MetricsMixin, PropertiesMixin, ProcessingMixin, ModelsMixin):

    PRE_FORWARD_ADAPTER: Callable = adapt_label_names_to_transformers
    POST_FORWARD_ADAPTER: Callable = None

    def __init__(self, hyperparameters):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.save_hyperparameters(hyperparameters)

        ModelsMixin.__init__(self)  # instantiates configs, models and tokenizer

        # process max sequence length
        if len(self.hyperparameters.max_sequence_length) == 1:
            self.hyperparameters.max_sequence_length = self.hyperparameters.max_sequence_length[0]

    def forward(self, *args, model: str = 'model', **kwargs):
        r""" Simply call the `model` attribute with the given args and kwargs. """
        assert not args, "positional arguments are deprecated, use only keyword arguments. "  # nosec

        kwargs = adapt_label_names_to_transformers(kwargs)

        assert hasattr(self, model), f"model does not have internal torch model {model}"  # nosec
        res = getattr(self, model)(**kwargs)

        if self.POST_FORWARD_ADAPTER is not None:
            res = self.__class__.POST_FORWARD_ADAPTER(res)
        return res

    def infer_batch_size(self, batch_size: Optional[int] = None) -> int:
        r""" Automatically get the batch size for the actual step. """
        new_batch_size = self.hyperparameters.batch_size

        if batch_size is not None and (new_batch_size != batch_size):
            raise ValueError(
                f"Inferred batch size {new_batch_size} is different "
                f"from provided batch size {batch_size}, check what batch size you are logging"
            )

        return new_batch_size

    def log(self, *args: List, batch_size: Optional[int] = None, kwargs: Dict = None):
        r""" Automatically manages logging:
        - Adds 'train/', 'valid/', 'test/' and prefixes
        - Logs single values, iterables and dicts
        - Adds correct batch size

        Args:
            *args: a single map between many names and values or a pair of name and value
            batch_size: the logging batch size, added automatically if not provided
            kwargs: additional kwargs to be passed to the original log function, added automatically if not provided
        """

        if kwargs is None:
            kwargs = DEFAULT_LOG_KWARGS[self.trainer.state.stage]

        # add stage prefix automatically
        data = parse_log_arguments(*args)
        data = {f"{TrainerStage_to_Names[self.trainer.state.stage]}/{name}": value for name, value in data.items()}
        batch_size = self.infer_batch_size(batch_size)

        # use super logger for synchronization and accumulation
        for k, v in data.items():
            super().log(k, v, batch_size=batch_size, **kwargs)

    def setup(self, *args, **kwargs):
        r""" Just check metrics are defined correctly. """
        MetricsMixin.setup(self, *args, **kwargs)

    def on_train_epoch_start(self):
        r""" Reset training metrics. """
        clean_device_cache()
        MetricsMixin.on_train_epoch_start(self)

    def on_validation_epoch_start(self):
        r""" Reset validation metrics. """
        clean_device_cache()
        MetricsMixin.on_validation_epoch_start(self)

    def on_test_epoch_start(self):
        r""" Reset test metrics. """
        clean_device_cache()
        MetricsMixin.on_test_epoch_start(self)

    def configure_optimizers(self):
        return OptimizersMixin.configure_optimizers(self)

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        MetricsMixin.add_argparse_args(parser)
        OptimizersMixin.add_argparse_args(parser)
        PropertiesMixin.add_argparse_args(parser)
        ProcessingMixin.add_argparse_args(parser)
        ModelsMixin.add_argparse_args(parser)


class ExtendedPipeline(Pipeline, ABC):
    r""" This class extended the base pipeline by adding support for extended tokenizers and multiple classification.
    This allow to encode more than 2 pieces of text together, for example in sequence/tuple/joint models, and to do
    multiple predictions for each input.
    """

    CONFIG_EXTENDED_CLASS: PretrainedConfig = None
    MODEL_EXTENDED_CLASS: PreTrainedModel = None
    TOKENIZER_EXTENDED_CLASS: ExtendedTokenizerFast = None

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        # checking and fixing classes
        if self.requires_extended_tokenizer():
            if self.TOKENIZER_EXTENDED_CLASS is None or not issubclass(
                self.TOKENIZER_EXTENDED_CLASS, ExtendedTokenizerFast
            ):
                raise ValueError(
                    "to encode the multiple inputs, you need to specify a `TOKENIZER_EXTENDED_CLASS`"
                )

        if self.requires_extended_model():
            # not requiring extended tokenizer because all k instances may also be encoded together as a single part
            if self.MODEL_EXTENDED_CLASS is None or self.CONFIG_EXTENDED_CLASS is None:
                raise ValueError(
                    "`CONFIG_EXTENDED_CLASS` and `MODEL_EXTENDED_CLASS` are required"
                )

    def configure_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        r""" Load or create the configuration and return it. """
        if self.requires_extended_model():
            self.CONFIG_CLASS = self.CONFIG_EXTENDED_CLASS
        if self.hyperparameters.k is not None:
            kwargs['k'] = self.hyperparameters.k
        if self.hyperparameters.extended_token_type_ids is not None:
            kwargs['type_vocab_size'] = max(self.hyperparameters.extended_token_type_ids) + 1
        return super().configure_config(**kwargs)

    def configure_model(self, config: PretrainedConfig, **kwargs) -> Union[PreTrainedModel, Dict[str, PreTrainedModel]]:
        if self.requires_extended_model():
            self.MODEL_CLASS = self.MODEL_EXTENDED_CLASS
        return super().configure_model(config, **kwargs)

    def configure_tokenizer(self, **kwargs) -> PreTrainedTokenizerBase:
        if self.requires_extended_tokenizer():
            self.TOKENIZER_CLASS = self.TOKENIZER_EXTENDED_CLASS
        return super().configure_tokenizer(**kwargs)

    @abstractmethod
    def requires_extended_tokenizer(self):
        r""" Return true if pipeline requires extended tokenizer for multiple sentence tokenization. """

    @abstractmethod
    def requires_extended_model(self):
        r""" Return true if pipeline requires extended model for multiple classification. """

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        super().add_argparse_args(parser)
        parser.add_argument(
            '--extended_token_type_ids',
            type=int,
            nargs='+',
            default=None,
            help="How many extended TT ids should be generated.",
        )
        parser.add_argument('-k', type=int, required=False, default=None)

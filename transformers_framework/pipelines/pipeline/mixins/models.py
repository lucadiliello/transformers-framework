import operator
from typing import Dict, Union

import torch
from lightning.pytorch.trainer import Trainer
from lightning_utilities.core.imports import compare_version
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase

from transformers_framework.utilities.arguments import FlexibleArgumentParser, parse_additional_kargs
from transformers_framework.utilities.functional import add_dict_to_attributes
from transformers_framework.utilities.logging import rank_zero_warn
from transformers_framework.utilities.models import load_config, load_model, load_tokenizer, set_decoder_start_token_id


_TORCH_GREATER_EQUAL_2 = compare_version("torch", operator.ge, "2.0.0", use_base_version=True)


class ModelsMixin:

    CONFIG_CLASS: PretrainedConfig = None
    MODEL_CLASS: PreTrainedModel = None
    TOKENIZER_CLASS: PreTrainedTokenizerBase = None

    def __init__(self):

        if self.hyperparameters.compile and not _TORCH_GREATER_EQUAL_2:
            raise ImportError(f"pytorch minimum version for compiling models is 2.0.0, got {torch.__version__}")

        self.fix_pre_trained_paths()

        # setup all configurations
        additiona_configuration_key_values = parse_additional_kargs(self.hyperparameters.additional_config_kwargs)
        configs = self.configure_config(**additiona_configuration_key_values)

        if not isinstance(configs, Dict):
            configs = {'config': configs}

        assert 'config' in configs, "Need to instantiate at least a config with key 'config'"  # nosec
        add_dict_to_attributes(self, configs)

        # instantiate model
        if not self.hyperparameters.configure_sharded_model:
            self.setup_models()

        # setup all configurations
        self.tokenizer = self.configure_tokenizer()

        # this simplifies our life with decoder models because every model defines start decoding token id differently
        if self.config.is_encoder_decoder or self.config.is_decoder:
            set_decoder_start_token_id(self.model, self.tokenizer)

    def setup_models(self):
        # setup all models
        models = self.configure_model(self.config)

        if not isinstance(models, Dict):
            models = {'model': models}

        # torch compile new functionality
        if self.hyperparameters.compile:
            models = {k: torch.compile(model) for k, model in models.items()}

        assert 'model' in models, "Need to instantiate at least a model with key 'model'"  # nosec
        add_dict_to_attributes(self, models)

    def configure_sharded_model(self):
        if self.hyperparameters.configure_sharded_model:
            self.setup_models()

    def fix_pre_trained_paths(self):
        r""" Checking and fixing pretrained paths. """
        if self.hyperparameters.pre_trained_config is None:
            self.hyperparameters.pre_trained_config = self.hyperparameters.pre_trained_model
            rank_zero_warn('Found None `pre_trained_config`, setting equal to `pre_trained_model`')

        if self.hyperparameters.pre_trained_tokenizer is None:
            self.hyperparameters.pre_trained_tokenizer = self.hyperparameters.pre_trained_model
            rank_zero_warn('Found None `pre_trained_tokenizer`, setting equal to `pre_trained_model`')

        if self.hyperparameters.pre_trained_config is None or self.hyperparameters.pre_trained_tokenizer is None:
            raise ValueError(
                "Cannot instantiate model without at least a `pre_trained_config` and a `pre_trained_tokenizer`"
            )

    def get_trainer_safe(self) -> Union[Trainer, None]:
        r""" Try to get trainer without raising exceptions. """
        try:
            return self.trainer
        except RuntimeError:
            return None

    def load_config(
        self, config_class: PretrainedConfig, name_or_path: str = None, **kwargs
    ) -> PretrainedConfig:
        r""" Load config from disk, HF or s3 automagically. """
        return load_config(
            config_class,
            name_or_path=name_or_path,
            temporary_models_folder=self.hyperparameters.temporary_models_folder,
            download_model_per_node=self.hyperparameters.prepare_data_per_node,
            trainer=self.get_trainer_safe(),
            **kwargs,
        )

    def load_model(
        self, model_class: PreTrainedModel, name_or_path: str = None, config: PretrainedConfig = None, **kwargs
    ) -> PreTrainedModel:
        r""" Load model from disk, HF or s3 automagically. """
        return load_model(
            model_class,
            name_or_path=name_or_path,
            config=config,
            temporary_models_folder=self.hyperparameters.temporary_models_folder,
            download_model_per_node=self.hyperparameters.prepare_data_per_node,
            trainer=self.get_trainer_safe(),
            **kwargs,
        )

    def load_tokenizer(
        self, tokenizer_class: PreTrainedTokenizerBase, name_or_path: str = None, **kwargs
    ) -> PreTrainedTokenizerBase:
        r""" Load tokenizer from disk, HF or s3 automagically. """
        return load_tokenizer(
            tokenizer_class,
            name_or_path=name_or_path,
            temporary_models_folder=self.hyperparameters.temporary_models_folder,
            download_model_per_node=self.hyperparameters.prepare_data_per_node,
            trainer=self.get_trainer_safe(),
            **kwargs,
        )

    def configure_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        r""" Load or create the configuration and return it. """
        return self.load_config(self.CONFIG_CLASS, name_or_path=self.hyperparameters.pre_trained_config, **kwargs)

    def configure_model(self, config: PretrainedConfig, **kwargs) -> Union[PreTrainedModel, Dict[str, PreTrainedModel]]:
        r"""
        Load model from scratch or from disk and return it. To load more than one model, for example in ELECTRA, you
        may return a dict of models.
        """
        return self.load_model(
            self.MODEL_CLASS, name_or_path=self.hyperparameters.pre_trained_model, config=config, **kwargs
        )

    def configure_tokenizer(self, **kwargs) -> PreTrainedTokenizerBase:
        r""" Load the tokenizer from disk and return it. """
        return self.load_tokenizer(
            self.TOKENIZER_CLASS, name_or_path=self.hyperparameters.pre_trained_tokenizer, **kwargs
        )

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):
        # add pre_trained model, tokenizer and config arguments. default config and tokenizer to model if missing
        parser.add_argument('--pre_trained_model', type=str, required=False, default=None)
        parser.add_argument('--pre_trained_tokenizer', type=str, required=False, default=None)
        parser.add_argument('--pre_trained_config', type=str, required=False, default=None)
        parser.add_argument(
            '--additional_config_kwargs',
            type=str,
            nargs='+',
            required=False,
            default=[],
            help="Additional key-values for the main config, to be passed as KEY=VALUE",
        )
        parser.add_argument('--compile', action="store_true", help="Compiles model graph for speedup")
        parser.add_argument(
            '--temporary_models_folder',
            type=str,
            required=False,
            default="/science/lucadiliello/.cache/transformers_framework/models",
        )
        parser.add_argument('--configure_sharded_model', action="store_true")

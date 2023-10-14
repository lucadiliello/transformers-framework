from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer.states import RunningStage
from torchmetrics import Metric
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers_framework.architectures.tokenization_utils import ExtendedTokenizerFast
from transformers_framework.interfaces.adaptation import adapt_label_names_to_transformers
from transformers_framework.optimizers import optimizers
from transformers_framework.schedulers import schedulers
from transformers_framework.utilities.arguments import FlexibleArgumentParser, parse_additional_kargs
from transformers_framework.utilities.collate import collate_flexible_numpy_fn
from transformers_framework.utilities.datamodules import TrainerFn, TrainerStage_to_Names
from transformers_framework.utilities.functional import add_dict_to_attributes, shrink_batch
from transformers_framework.utilities.logging import (
    parse_log_arguments,
    rank_zero_debug,
    rank_zero_info,
    rank_zero_warn,
)
from transformers_framework.utilities.methods import native
from transformers_framework.utilities.models import load_config, load_model, load_tokenizer, set_decoder_start_token_id
from transformers_framework.utilities.torch import clean_device_cache


DEFAULT_LOG_KWARGS = {
    RunningStage.TRAINING: dict(on_epoch=True, prog_bar=True, sync_dist=True),
    RunningStage.VALIDATING: dict(on_epoch=True, prog_bar=True, sync_dist=True),
    RunningStage.TESTING: dict(on_epoch=True, prog_bar=True, sync_dist=True),
    RunningStage.SANITY_CHECKING: dict(on_epoch=True, prog_bar=True, sync_dist=True),
}


class Pipeline(LightningModule):

    CONFIG_CLASS: PretrainedConfig = None
    MODEL_CLASS: PreTrainedModel = None
    TOKENIZER_CLASS: PreTrainedTokenizerBase = None

    PRE_FORWARD_ADAPTER: Callable = adapt_label_names_to_transformers
    POST_FORWARD_ADAPTER: Callable = None
    MODEL_INPUT_NAMES_TO_REDUCE: List[Tuple[str]] = None

    def __init__(self, hyperparameters):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.save_hyperparameters(hyperparameters)
        self.fix_pre_trained_paths()

    def configure_model(self):
        r""" All models initializations goes here. """
    
        # setup all configurations
        additiona_configuration_key_values = parse_additional_kargs(self.hyperparameters.additional_config_kwargs)
        configs = self.setup_config(**additiona_configuration_key_values)

        if not isinstance(configs, Dict):
            configs = {'config': configs}

        assert 'config' in configs, "Need to instantiate at least a config with key 'config'"  # nosec
        add_dict_to_attributes(self, configs)

        # setup all models
        models = self.setup_model()
        models = {'model': models} if not isinstance(models, Dict) else models  # eventually convert to dict

        # torch compile new functionality
        if self.hyperparameters.compile:
            models = {k: torch.compile(model) for k, model in models.items()}

        assert 'model' in models, "Need to instantiate at least a model with key 'model'"  # nosec
        add_dict_to_attributes(self, models)

        # setup all configurations
        self.tokenizer = self.setup_tokenizer()

        # this simplifies our life with decoder models because every model defines start decoding token id differently
        if self.config.is_encoder_decoder or self.config.is_decoder:
            set_decoder_start_token_id(self.model, self.tokenizer)

    def forward(self, model: str = 'model', **kwargs):
        r""" Forward pass with pre and post processing. """

        if self.PRE_FORWARD_ADAPTER is not None:
            kwargs = self.__class__.PRE_FORWARD_ADAPTER(kwargs)

        assert hasattr(self, model), f"model does not have internal torch model {model}"  # nosec
        res = getattr(self, model)(**kwargs)

        if self.POST_FORWARD_ADAPTER is not None:
            res = self.__class__.POST_FORWARD_ADAPTER(res)

        return res

    ###############################################################################################
    # MODELS ######################################################################################
    ###############################################################################################

    def fix_pre_trained_paths(self):
        r""" Fix pre-training paths of models. """

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

    def setup_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        r""" Load or create the configuration and return it. """
        config_class = kwargs.pop('config_class', self.CONFIG_CLASS)
        pre_trained_config = kwargs.pop('pre_trained_config', self.hyperparameters.pre_trained_config)
        temporary_models_folder = kwargs.pop('temporary_models_folder', self.hyperparameters.temporary_models_folder)
        prepare_data_per_node = kwargs.pop('prepare_data_per_node', self.hyperparameters.prepare_data_per_node)

        return load_config(
            config_class=config_class,
            name_or_path=pre_trained_config,
            temporary_models_folder=temporary_models_folder,
            download_model_per_node=prepare_data_per_node,
            trainer=self.trainer,
            **kwargs,
        )

    def setup_model(self, **kwargs) -> Union[PreTrainedModel, Dict[str, PreTrainedModel]]:
        r"""
        Load model from scratch or from disk and return it. To load more than one model, for example in ELECTRA, you
        may return a dict of models.
        """
        config = kwargs.pop('config', self.config)
        model_class = kwargs.pop('model_class', self.MODEL_CLASS)
        pre_trained_model = kwargs.pop('pre_trained_model', self.hyperparameters.pre_trained_model)
        temporary_models_folder = kwargs.pop('temporary_models_folder', self.hyperparameters.temporary_models_folder)
        prepare_data_per_node = kwargs.pop('prepare_data_per_node', self.hyperparameters.prepare_data_per_node)

        return load_model(
            model_class=model_class,
            name_or_path=pre_trained_model,
            config=config,
            temporary_models_folder=temporary_models_folder,
            download_model_per_node=prepare_data_per_node,
            trainer=self.trainer,
            **kwargs,
        )

    def setup_tokenizer(self, **kwargs) -> PreTrainedTokenizerBase:
        r""" Load the tokenizer from disk and return it. """
        tokenizer_class = kwargs.pop('tokenizer_class', self.TOKENIZER_CLASS)
        pre_trained_tokenizer = kwargs.pop('pre_trained_tokenizer', self.hyperparameters.pre_trained_tokenizer)
        temporary_models_folder = kwargs.pop('temporary_models_folder', self.hyperparameters.temporary_models_folder)
        prepare_data_per_node = kwargs.pop('prepare_data_per_node', self.hyperparameters.prepare_data_per_node)

        return load_tokenizer(
            tokenizer_class=tokenizer_class,
            name_or_path=pre_trained_tokenizer,
            temporary_models_folder=temporary_models_folder,
            download_model_per_node=prepare_data_per_node,
            trainer=self.trainer,
            **kwargs,
        )

    ###############################################################################################
    # OPTIMIZATION ################################################################################
    ###############################################################################################

    def num_training_steps(self) -> int:
        r""" Total training steps inferred from datasets length, number of nodes and devices. """
        return (
            self.hyperparameters.max_steps
            if self.hyperparameters.max_steps is not None and self.hyperparameters.max_steps > 0
            else self.trainer.estimated_stepping_batches
        )

    def configure_optimizers(self):
        r""" Instantiate an optimizer on the parameters of self.
        A scheduler is also instantiated to manage the learning rate.
        """
        optimizer_class = optimizers[self.hyperparameters.optimizer]
        optimizer = optimizer_class(self.hyperparameters, self.named_parameters())
    
        scheduler_class = schedulers[self.hyperparameters.scheduler]
        scheduler = scheduler_class(self.hyperparameters, optimizer, self.num_training_steps())

        return {
            'optimizer': optimizer,
            'lr_scheduler':
                {
                    'scheduler': scheduler,  # The LR schduler
                    'interval': self.hyperparameters.scheduler_interval,  # The unit of the scheduler's step size
                    'frequency': self.hyperparameters.scheduler_frequency,  # The frequency of the scheduler
                }
        }

    ###############################################################################################
    # METRICS #####################################################################################
    ###############################################################################################
        
    def reset_metrics(self, stage: str):
        r""" Reset all metrics in model for the given stage. """
        rank_zero_debug(f"Resetting {stage} metrics")
        for name, module in self.named_children():
            if isinstance(module, Metric) and name.startswith(stage):
                module.reset()

    def setup(self, stage: str):
        r""" Just check metrics are defined correctly. """
        rank_zero_info(f"Running setup for stage {stage}")

        for name, module in self.named_children():
            if isinstance(module, Metric):
                if not (
                    name.startswith('train')
                    or name.startswith('valid')
                    or name.startswith('test')
                ):
                    raise ValueError("All metrics in the model must start with 'train', 'valid' or 'test' ")

    def on_train_epoch_start(self):
        r""" Reset training metrics. """
        clean_device_cache()
        self.reset_metrics('train')

    def on_validation_epoch_start(self):
        r""" Reset validation metrics. """
        clean_device_cache()
        self.reset_metrics('valid')

    def on_test_epoch_start(self):
        r""" Reset test metrics. """
        clean_device_cache()
        self.reset_metrics('test')

    ###############################################################################################
    # DATA ########################################################################################
    ###############################################################################################

    @native
    def preprocess(
        self,
        dataset: Dataset,
        num_workers: int = None,
        batch_size: int = None,
        load_from_cache_file: bool = True,
        **kwargs: Dict[str, Any],
    ) -> Dataset:
        r""" Preprocess a dataset. Useful for tasks like MR, NER and POS. """
        ...

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        r""" Postprocess a single sample. Samples will be extracted from the dataset returned above.
        This operation will be performed in dataloader's workers.
        Please use numpy as much as you can to speed up operations.
        """
        return sample

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        r""" Merge together samples drawn from dataset after `postprocess`. This method should return
        numpy arrays of torch tensors. This is a good place for padding and shrinking tensors. """
        batch = [self.postprocess(sample) for sample in batch]
        batch = collate_flexible_numpy_fn(batch)

        if self.MODEL_INPUT_NAMES_TO_REDUCE:
            for group in self.MODEL_INPUT_NAMES_TO_REDUCE:
                if group[0] not in batch or batch[group[0]] is None:
                    raise ValueError(
                        f"requested to reduce {group} but batch does not contain"
                        f"at least the first key {group[0]} or is None. available keys in batch: {tuple(batch.keys())}"
                    )
                self.shrink_batch(batch, group)
        return batch

    def shrink_batch(self, batch: Dict[str, Any], keys: List[str] = None, pad_token_id: int = None):
        r""" Remove data on the sequence length dimension in the positions where every example is padded. """
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id

        # if self.hyperparameters.compile:
        #     return

        # remove from group keys that are not available in the batch or that are None
        keys = [keys[0]] + list(set(keys[1:]).intersection(batch.keys()))
        keys = [k for k in keys if batch[k] is not None]

        shrink_batch(
            batch=batch,
            keys=keys,
            pad_token_id=pad_token_id,
            shrink_to_multiples_of=(
                8 if self.trainer.precision in ("16-true", "16-mixed", "bf16-true", "bf16-mixed") else None
            ),
        )

    ##############
    # PROPERTIES #
    ##############

    @property
    def is_training(self):
        return self.trainer.state.fn == TrainerFn.FITTING

    @property
    def is_validating(self):
        return self.trainer.state.fn == TrainerFn.VALIDATING

    @property
    def is_testing(self):
        return self.trainer.state.fn == TrainerFn.TESTING

    @property
    def is_predicting(self):
        return self.trainer.state.fn == TrainerFn.PREDICTING

    #############
    # UTILITIES #
    #############

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

    @classmethod
    def add_argparse_args(cls, parser: FlexibleArgumentParser):

        # Data
        parser.add_argument(
            '--max_sequence_length', type=int, nargs="+", required=True, help="Input max sequence length(s)"
        )

        # Modeling
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

        # Optimization
        parser.add_argument('--optimizer', type=str, default='adamw', choices=optimizers.keys())
        parser.add_argument('--scheduler', type=str, default='linear_decay', choices=schedulers.keys())
        parser.add_argument('--scheduler_interval', type=str, default='step', choices=('step', 'epoch'))
        parser.add_argument('--scheduler_frequency', type=int, default=1)

        # retrieving classes with temporary parsered arguments
        tmp_params, _ = parser.parse_known_args()

        # get pl_model_class in advance to know which params it needs
        optimizers[tmp_params.optimizer].add_argparse_args(parser)
        schedulers[tmp_params.scheduler].add_argparse_args(parser)


class ExtendedPipeline(Pipeline):
    r""" This class extended the base pipeline by adding support for extended tokenizers and multiple classification.
    This allow to encode more than 2 pieces of text together, for example in sequence/tuple/joint models, and to do
    multiple predictions for each input.
    """

    CONFIG_EXTENDED_CLASS: PretrainedConfig = None
    MODEL_EXTENDED_CLASS: PreTrainedModel = None
    TOKENIZER_EXTENDED_CLASS: ExtendedTokenizerFast = None

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        assert not self.hyperparameters.pad_to_k or self.hyperparameters.k is not None, (
            "Cannot set `--pad_to_k` when k is None"
        )

        # checking and fixing classes
        if self.requires_extended_tokenizer:
            if self.TOKENIZER_EXTENDED_CLASS is None or not issubclass(
                self.TOKENIZER_EXTENDED_CLASS, ExtendedTokenizerFast
            ):
                raise ValueError(
                    "to encode the multiple inputs, you need to specify a `TOKENIZER_EXTENDED_CLASS`"
                )

        if self.requires_extended_model:
            # not requiring extended tokenizer because all k instances may also be encoded together as a single part
            if self.MODEL_EXTENDED_CLASS is None or self.CONFIG_EXTENDED_CLASS is None:
                raise ValueError(
                    "`CONFIG_EXTENDED_CLASS` and `MODEL_EXTENDED_CLASS` are required"
                )

    def setup_config(self, **kwargs) -> Union[PretrainedConfig, Dict[str, PretrainedConfig]]:
        r""" Load or create the configuration and return it. """
        if self.requires_extended_model:
            kwargs['config_class'] = kwargs.pop('config_class', self.CONFIG_EXTENDED_CLASS)
            kwargs['k'] = kwargs.pop('k', self.hyperparameters.k)
        if self.requires_extended_token_type_ids:
            num_token_type_ids = max(self.hyperparameters.extended_token_type_ids) + 1
            kwargs['type_vocab_size'] = kwargs.pop('type_vocab_size', num_token_type_ids)
        return super().setup_config(**kwargs)

    def setup_model(self, **kwargs) -> Union[PreTrainedModel, Dict[str, PreTrainedModel]]:
        r""" Load or create the model and return it. """
        if self.requires_extended_model:
            kwargs['model_class'] = kwargs.pop('model_class', self.MODEL_EXTENDED_CLASS)
        return super().setup_model(**kwargs)

    def setup_tokenizer(self, **kwargs) -> PreTrainedTokenizerBase:
        r""" Load or create the tokenizer and return it. """
        if self.requires_extended_tokenizer:
            kwargs['tokenizer_class'] = kwargs.pop('tokenizer_class', self.TOKENIZER_EXTENDED_CLASS)
        return super().setup_tokenizer(**kwargs)

    @property
    def requires_extended_tokenizer(self):
        r""" Return true if pipeline requires extended tokenizer for multiple sentence tokenization. """
        return len(self.hyperparameters.input_columns) > 2 or self.requires_extended_model

    @property
    def requires_extended_token_type_ids(self):
        return self.hyperparameters.extended_token_type_ids is not None

    @property
    def requires_extended_model(self):
        r""" Return true if pipeline requires extended model for multiple classification. """
        return self.hyperparameters.k is not None

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
        parser.add_argument('--pad_to_k', action="store_true")

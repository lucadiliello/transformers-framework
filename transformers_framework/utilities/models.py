import math
import os
from typing import Dict, List, Tuple

import boto3
import torch
import torch.distributed as dist
from lightning.pytorch.trainer import Trainer
from torch import nn
from tqdm import tqdm
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.models.longt5.modeling_longt5 import _split_into_blocks
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers_framework.utilities.hash import hash_string
from transformers_framework.utilities.logging import rank_zero_info, rank_zero_warn


def tie_or_clone_weights(output_embeddings, input_embeddings):
    r"""Tie or clone module weights and optionally biases if they are present. """
    if output_embeddings.weight.shape != input_embeddings.weight.shape:
        raise ValueError(
            f"Cannot tie weights, size mismatch: {output_embeddings.weight.shape} vs {input_embeddings.weight.shape}"
        )
    output_embeddings.weight = input_embeddings.weight

    if hasattr(output_embeddings, "bias") and hasattr(input_embeddings, "bias"):
        output_embeddings.bias = input_embeddings.bias

    if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
        output_embeddings.out_features = input_embeddings.num_embeddings


def tie_weights_electra(
    generator, discriminator, tie_generator_discriminator_embeddings: bool, tie_word_embeddings: bool
):
    r"""
    Tie the weights between the generator and the discriminator embeddings.
    Electra paper says to link both the token and the positional embeddings, see Section 3.2
    """
    if tie_generator_discriminator_embeddings:
        # token embeddings
        tie_or_clone_weights(
            discriminator.electra.embeddings.word_embeddings,
            generator.electra.embeddings.word_embeddings
        )

        # positional embeddings
        tie_or_clone_weights(
            discriminator.electra.embeddings.position_embeddings,
            generator.electra.embeddings.position_embeddings
        )

        # token type embeddings
        tie_or_clone_weights(
            discriminator.electra.embeddings.token_type_embeddings,
            generator.electra.embeddings.token_type_embeddings
        )

        # layernorm weights
        tie_or_clone_weights(
            discriminator.electra.embeddings.LayerNorm, generator.electra.embeddings.LayerNorm
        )

    # assert all weights are tied
    if tie_word_embeddings:
        assert generator.generator_lm_head.weight is generator.electra.embeddings.word_embeddings.weight

    if tie_generator_discriminator_embeddings:
        assert generator.generator_lm_head.weight is (
            discriminator.electra.embeddings.word_embeddings.weight
        )
        assert generator.electra.embeddings.word_embeddings.weight is (
            discriminator.electra.embeddings.word_embeddings.weight
        )
        assert generator.electra.embeddings.position_embeddings.weight is (
            discriminator.electra.embeddings.position_embeddings.weight
        )
        assert generator.electra.embeddings.token_type_embeddings.weight is (
            discriminator.electra.embeddings.token_type_embeddings.weight
        )
        assert generator.electra.embeddings.LayerNorm.weight is (
            discriminator.electra.embeddings.LayerNorm.weight
        )


def tie_weights_deberta(generator, discriminator, tie_generator_discriminator_embeddings: bool):
    r""" DeBERTa does not directly tie the embedding weights of the generator and the discriminator. """

    # token embeddings
    if tie_generator_discriminator_embeddings:
        tie_or_clone_weights(
            discriminator.deberta.embeddings.word_embeddings,
            generator.deberta.embeddings.word_embeddings
        )

        assert generator.get_output_embeddings().weight is (
            discriminator.deberta.embeddings.word_embeddings.weight
        )
        assert generator.deberta.embeddings.word_embeddings.weight is (
            discriminator.deberta.embeddings.word_embeddings.weight
        )

        # positional embeddings
        if (
            discriminator.deberta.embeddings.position_embeddings is not None
            and generator.deberta.embeddings.position_embeddings is not None
        ):
            tie_or_clone_weights(
                discriminator.deberta.embeddings.position_embeddings,
                generator.deberta.embeddings.position_embeddings
            )

            assert generator.deberta.embeddings.position_embeddings.weight is (
                discriminator.deberta.embeddings.position_embeddings.weight
            )
            
        if (
            hasattr(discriminator.deberta.embeddings, "token_type_embeddings")
            and hasattr(generator.deberta.embeddings, "token_type_embeddings")
        ):
            # token type embeddings
            tie_or_clone_weights(
                discriminator.deberta.embeddings.token_type_embeddings,
                generator.deberta.embeddings.token_type_embeddings
            )

            assert generator.deberta.embeddings.token_type_embeddings.weight is (
                discriminator.deberta.embeddings.token_type_embeddings.weight
            )

        # layernorm weights
        tie_or_clone_weights(
            discriminator.deberta.embeddings.LayerNorm,
            generator.deberta.embeddings.LayerNorm
        )

        assert generator.deberta.embeddings.LayerNorm.weight is (
            discriminator.deberta.embeddings.LayerNorm.weight
        )

        if (
            hasattr(discriminator.deberta.embeddings, "embed_proj")
            and hasattr(generator.deberta.embeddings, "embed_proj")
        ):
            # proj weights
            tie_or_clone_weights(
                discriminator.deberta.embeddings.embed_proj,
                generator.deberta.embeddings.embed_proj
            )
        
            assert generator.deberta.embeddings.embed_proj.weight is (
                discriminator.deberta.embeddings.embed_proj.weight
            )


def get_electra_reduced_generator_config(discriminator_config: ElectraConfig, factor: float = 1 / 3, **kwargs):
    r""" Created reduced configuration for electra generator. """
    params = {
        **vars(discriminator_config),
        'hidden_size': int(discriminator_config.hidden_size * factor),
        'num_attention_heads': int(discriminator_config.num_attention_heads * factor),
        'intermediate_size': int(discriminator_config.intermediate_size * factor),
        **kwargs,
    }
    return ElectraConfig(**params)


def get_deberta_reduced_generator_config(discriminator_config: DebertaV2Config, factor: float = 1 / 2, **kwargs):
    r""" Created reduced configuration for deberta generator. """
    params = {
        **vars(discriminator_config),
        'num_hidden_layers': int(discriminator_config.num_hidden_layers * factor),
        **kwargs,
    }
    return DebertaV2Config(**params)


class DownloadProgress(object):

    def __init__(self, total: int = None, name: str = None):
        super().__init__()
        name = "Downloading" if name is None else f"Downloading {os.path.split(name)[-1]}"
        self.progress = tqdm(desc=name, total=total, unit="B", unit_scale=True)

    def __enter__(self):
        return self.callback

    def callback(self, chunk):
        self.progress.update(chunk)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.close()


def download_s3_folder(s3_path: str, local_dir: str):
    r"""
    Download the contents of a folder recursively into a directory

    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """

    assert s3_path.startswith("s3://")
    s3_path = s3_path.removeprefix("s3://")

    bucket_name, *path_parts = s3_path.split(os.sep)
    s3_folder = os.path.join(*path_parts)

    # assumes credentials & configuration are handled outside python in .aws directory or environment variables
    s3_resource = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bucket = s3_resource.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))

        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue

        # getting metadata for progress bar
        meta_data = s3_client.head_object(Bucket=bucket.name, Key=obj.key)
        total_length = int(meta_data.get('ContentLength', 0))

        with DownloadProgress(total=total_length, name=obj.key) as callback:
            bucket.download_file(obj.key, target, Callback=callback)

    rank_zero_info("Syncronization from s3 completed successfully")


def download_model_from_s3(
    model_path: str, temporary_models_folder: str = None, download_model_per_node: bool = True, trainer: Trainer = None
) -> str:
    r""" Guard to run code only on local/global (based on download_model_per_node) rank in distributed setting. """

    os.makedirs(temporary_models_folder, exist_ok=True)

    def should_download():
        if dist.is_available() and dist.is_initialized() and trainer is not None:
            if download_model_per_node:
                return trainer.local_rank == 0
            else:
                return trainer.global_rank == 0
        return True

    model_name_hash = hash_string(model_path)
    model_path_with_hash = os.path.join(temporary_models_folder, model_name_hash)

    if not os.path.isdir(model_path_with_hash) and should_download():
        rank_zero_info(f"Syncronizing s3 path {model_path} to local folder {model_path_with_hash}")
        download_s3_folder(model_path, model_path_with_hash)

    if trainer is not None:
        trainer.strategy.barrier()
    return model_path_with_hash


def init_mismatched_keys(model: PreTrainedModel, info: Dict[str, List]):
    r"""
    Initialize the modules that were not initialized while loading the ckpt because of wrong shape.
    Args:
        model: transformers model that was initialized from checkpoint
        info: dict returned by `model.from_pretrained(..., output_loading_info=True)`
    """

    loaded_keys = [k[0] for k in info['mismatched_keys']]

    expected_keys = list(model.state_dict().keys())
    prefix = model.base_model_prefix

    if len(prefix) > 0:
        has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
        expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
    else:
        has_prefix_module = False
        expects_prefix_module = False

    # key re-naming operations are never done on the keys
    # that are loaded, but always on the keys of the newly initialized model
    remove_prefix_from_model = not has_prefix_module and expects_prefix_module
    add_prefix_to_model = has_prefix_module and not expects_prefix_module

    modules = model.retrieve_modules_from_names(
        loaded_keys,
        add_prefix=add_prefix_to_model,
        remove_prefix=remove_prefix_from_model
    )

    for (key, old_size, new_size), module in zip(info['mismatched_keys'], modules):
        rank_zero_warn(f'Initializing mismatched weights {key}, size {old_size} -> {new_size}')
        model._init_weights(module)


def load_pretrained_safe(
    model_class: PreTrainedModel, name_or_path: str, config: PretrainedConfig = None, **kwargs
) -> PreTrainedModel:
    r""" Load a model in safe environment, taking care of initializing eventual parameters
    that were not loaded from the checkpoint. """

    model, info = model_class.from_pretrained(
        name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
        **kwargs,
    )
    init_mismatched_keys(model, info)
    return model


def load_config(
    config_class: PretrainedConfig,
    name_or_path: str = None,
    temporary_models_folder: str = None,
    download_model_per_node: bool = True,
    trainer: Trainer = None,
    **kwargs
):
    r""" Load a config in safe environment.
    If the `name_or_path` is None, the model will be created safely from scratch. """

    if name_or_path is not None:
        if name_or_path.startswith('s3://'):
            assert temporary_models_folder is not None

            name_or_path = download_model_from_s3(
                name_or_path,
                temporary_models_folder=temporary_models_folder,
                download_model_per_node=download_model_per_node,
                trainer=trainer,
            )

        config = config_class.from_pretrained(name_or_path, **kwargs)
    else:
        rank_zero_warn(f"Config {config_class.__name__} loaded from scratch and not from a pretrained ckpt.")
        config = config_class(**kwargs)

    return config


def load_model(
    model_class: PreTrainedModel,
    name_or_path: str = None,
    config: PretrainedConfig = None,
    temporary_models_folder: str = None,
    download_model_per_node: bool = True,
    trainer: Trainer = None,
    **kwargs
):
    r""" Load a model in safe environment, taking care of initializing eventual parameters
    that were not loaded from the checkpoint. If the `name_or_path` is None, the model will be
    created safely from scratch. """

    if name_or_path is not None:
        if name_or_path.startswith('s3://'):
            assert temporary_models_folder is not None

            name_or_path = download_model_from_s3(
                name_or_path,
                temporary_models_folder=temporary_models_folder,
                download_model_per_node=download_model_per_node,
                trainer=trainer,
            )

        model = load_pretrained_safe(model_class, name_or_path, config=config, **kwargs)
    else:
        rank_zero_warn(f"Model {model_class.__name__} loaded from scratch and not from a pretrained model.")
        model = model_class(config, **kwargs)

    return model


def load_tokenizer(
    tokenizer_class: PretrainedConfig,
    name_or_path: str = None,
    temporary_models_folder: str = None,
    download_model_per_node: bool = True,
    trainer: Trainer = None,
    **kwargs
):
    r""" Load a tokenizer in safe environment.
    If the `name_or_path` is None, the model will be created safely from scratch. """

    if name_or_path is not None:
        if name_or_path.startswith('s3://'):
            assert temporary_models_folder is not None

            name_or_path = download_model_from_s3(
                name_or_path,
                temporary_models_folder=temporary_models_folder,
                download_model_per_node=download_model_per_node,
                trainer=trainer,
            )

        tokenizer = tokenizer_class.from_pretrained(name_or_path, **kwargs)
    else:
        rank_zero_warn(f"Config {tokenizer_class.__name__} loaded from scratch and not from a pretrained ckpt.")
        tokenizer = tokenizer_class(**kwargs)

    return tokenizer


def compute_relative_position_bucket(
    relative_position: torch.Tensor, num_buckets: int = 32, max_distance: int = 128
):
    r"""
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. We use smaller buckets for small absolute relative_position and larger buckets for larger absolute
    relative_positions. All relative positions >= max_distance map to the same bucket.
    All relative positions <= -max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on.

    Args:
        relative_position: an int32 Tensor
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    if num_buckets <= 3:
        raise ValueError(f"`num_buckets` must be greater than 3, got {num_buckets}")

    # buckets on left and right
    num_buckets //= 2

    relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
    relative_position = torch.abs(relative_position)
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)

    # compute the upper bound
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(relative_position < max_exact, relative_position, relative_position_if_large)
    return relative_buckets


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    r"""
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


def create_position_ids_from_inputs_embeds(inputs_embeds, padding_idx):
    r"""
    We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

    Args:
        inputs_embeds: torch.Tensor

    Returns: torch.Tensor
    """
    input_shape = inputs_embeds.size()[:-1]
    sequence_length = input_shape[1]

    position_ids = torch.arange(
        padding_idx + 1, sequence_length + padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
    )
    return position_ids.unsqueeze(0).expand(input_shape)


def make_global_fixed_block_ids(attention_mask: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r""" Obtain the "segment block ids" corresponding to each input block. """
    batch_size, seq_len = attention_mask.shape[:2]

    block_ids = torch.arange(0, seq_len, device=attention_mask.device).unsqueeze(0).repeat(batch_size, 1) / block_size
    block_ids = torch.floor(block_ids) * attention_mask + (attention_mask - 1)

    if seq_len % block_size != 0:
        attention_mask = nn.functional.pad(
            attention_mask, pad=(0, block_size - (seq_len % block_size), 0, 0), value=-1
        )

    attention_mask = attention_mask.reshape(batch_size, -1, block_size).max(dim=-1).values
    return block_ids, attention_mask


def make_side_relative_position_ids(attention_mask: torch.Tensor, block_size: int) -> torch.Tensor:
    r""" Create the relative position tensor for local -> global attention. """
    block_ids, global_segment_ids = make_global_fixed_block_ids(attention_mask, block_size)

    global_positions = torch.arange(global_segment_ids.shape[-1], device=block_ids.device)
    side_relative_position = global_positions - block_ids[..., None]

    return side_relative_position.type(torch.int64)


def get_summary_attention_mask(attention_mask: torch.Tensor, block_len: int, device: torch.device) -> torch.Tensor:
    r""" Prepare attention mask to be applied for a local attention. """

    # [batch_size, num_blocks, block_len]
    _blocked_attention_mask = _split_into_blocks(attention_mask, block_len, dim=1)

    # unsqueeze on different dims to ensure output has following shape
    local_attention_mask = torch.logical_and(
        _blocked_attention_mask.unsqueeze(-1),
        _blocked_attention_mask.unsqueeze(-2),
    )

    # [batch_size, 1, num_block, block_len, block_len]
    return local_attention_mask.unsqueeze(1).to(device)


def set_decoder_start_token_id(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
    r""" Set decoder_start_token_id in tokenizer. """

    decoder_start_token_id = model._get_decoder_start_token_id()
    setattr(tokenizer, "decoder_start_token_id", decoder_start_token_id)

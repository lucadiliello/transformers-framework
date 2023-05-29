from argparse import Namespace
from typing import Union

import torch
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.strategy import Strategy

from transformers_framework.utilities.logging import rank_zero_error, rank_zero_info, rank_zero_warn


def initialize_strategy(hyperparameters: Namespace) -> Union[Strategy, str]:
    r""" Check if particular strategy is used and eventually instantiate directly. """

    # disable find unused parameters to improve performance
    if hyperparameters.strategy in ("dp", "ddp2", "ddp_spawn"):
        rank_zero_error(
            "This repo is not designed to work with DataParallel or spawn strategy."
            "Use strategy `ddp` or others instead."
        )
        exit(1)

    if hyperparameters.strategy == "ddp":
        return DDPStrategy(find_unused_parameters=False)

    elif hyperparameters.strategy == "colossalai":
        from lightning_colossalai import ColossalAIStrategy
        return ColossalAIStrategy(
            use_chunk=False,
            enable_distributed_storage=True,
            placement_policy='cuda',
            initial_scale=32,
        )

    elif hyperparameters.strategy == 'deepspeed_stage_2_offload_no_error':
        import deepspeed
        from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy

        from transformers_framework.utilities.deepspeed import get_dynamic_loss_scale_args

        deepspeed.runtime.config.get_dynamic_loss_scale_args = get_dynamic_loss_scale_args

        return DeepSpeedStrategy(
            stage=2,
            offload_optimizer=False,
            offload_parameters=False,
            config={
                "fp16": {
                    "enabled": True,
                    "loss_scale": 2.0,
                    "initial_scale_power": 32,
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                    "min_loss_scale": 0.1,
                    "raise_error_at_min_scale": False,
                }
            }
        )

    elif hyperparameters.strategy == 'deepspeed_stage_3_offload':
        import deepspeed
        from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy

        return DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
        )

    return hyperparameters.strategy


def initialize_precision(hyperparameters: Namespace) -> Union[str, int]:
    r""" Checks over precision and used model. """

    # no gradient clipping needed if running with fp16
    if hyperparameters.precision == 16 and hyperparameters.gradient_clip_val:
        rank_zero_warn(
            "There is no need to use `gradient_clip_val` with `precision=16` since gradients are scaled"
            " automatically before the optimization step."
        )

    # precision speedup
    rank_zero_info("Activating fast FP32 matmul")
    torch.set_float32_matmul_precision('medium')

    if hyperparameters.accelerator == "gpu" and hyperparameters.precision not in (16, 'bf16'):
        rank_zero_info("Activating fast TF32 matmul")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if 't5' in hyperparameters.model and hyperparameters.precision == 16 and (
        'accelerator' not in hyperparameters or hyperparameters.accelerator != 'mps'
    ):
        rank_zero_warn("Precision set to 16 (FP16) but model is T5-like. Changing precision to bf16 for you.")
        return 'bf16'

    return hyperparameters.precision

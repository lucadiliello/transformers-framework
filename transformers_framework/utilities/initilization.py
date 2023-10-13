import os
from transformers_framework.utilities.classes import ExtendedNamespace
from typing import Union

import torch
from lightning.pytorch.profilers.profiler import Profiler
from lightning.pytorch.profilers.pytorch import PyTorchProfiler
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.strategies.strategy import Strategy
from lightning.pytorch.plugins import BitsandbytesPrecisionPlugin

from transformers_framework.utilities.logging import rank_zero_error, rank_zero_info, rank_zero_warn


def initialize_strategy(hyperparameters: ExtendedNamespace) -> Union[Strategy, str]:
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

    elif hyperparameters.strategy == 'deepspeed_stage_2_offload_no_error':
        import deepspeed
        from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy

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
        from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy

        return DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
        )

    return hyperparameters.strategy


def initialize_precision(hyperparameters: ExtendedNamespace) -> Union[str, int]:
    r""" Checks over precision and used model. """

    precision = hyperparameters.precision

    # no gradient clipping needed if running with fp16
    if precision in ('16-mixed', '16-true') and hyperparameters.gradient_clip_val:
        rank_zero_warn(
            "There is no need to use `gradient_clip_val` with `precision=16-*` "
            "since gradients are scaled automatically before the optimization step."
        )

    # precision speedup
    fast_fp32_matmul = os.environ.get('TRANSFORMERS_FRAMEWORK_FP32_MATMUL_PRECISION', 'medium')
    rank_zero_info(f"Using FP32 matmul: {fast_fp32_matmul}")
    torch.set_float32_matmul_precision(fast_fp32_matmul)

    if hyperparameters.accelerator == "gpu":
        cudnn_enabled = eval(os.environ.get('TRANSFORMERS_FRAMEWORK_CUDNN_ENABLED', 'True'))
        allow_tf32 = eval(os.environ.get('TRANSFORMERS_FRAMEWORK_TF32_ENABLED', 'True'))

        rank_zero_info(f"Using CUDNN: {cudnn_enabled}")
        rank_zero_info(f"Using fast TF32 matmul: {allow_tf32}")

        torch.backends.cudnn.enabled = cudnn_enabled
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    # bitsandbytes
    if precision in ('nf4', 'f4-dq', 'fp4', 'fp4-dq', 'int8', 'int8-training'):
        return BitsandbytesPrecisionPlugin(precision)

    if 't5' in hyperparameters.model:
        if precision == '16-mixed' and hyperparameters.get('accelerator') != 'mps':
            rank_zero_warn("Precision set to '16-mixed' but model is T5. Changing precision to 'bf16-mixed' for you.")
            return 'bf16-mixed'

        if precision == '16-true' and hyperparameters.get('accelerator') != 'mps':
            rank_zero_warn("Precision set to '16-true' but model is T5. Changing precision to 'bf16-true' for you.")
            return 'bf16-true'

    return precision


def initialize_profiler(hyperparameters: ExtendedNamespace) -> Union[str, Profiler]:
    r""" Initialize profiler in the case advanced GPU checks are required. """
    if hyperparameters.profiler is None or hyperparameters.profiler != 'cpu_gpu_memory':
        return hyperparameters.profiler

    return PyTorchProfiler(
        dirpath=None,
        filename='pytorch_cpu_gpu_memory_report',
        group_by_input_shapes=True,
        emit_nvtx=False,
        export_to_chrome=False,
        record_module_names=True,
        row_limit=-1,
        record_shapes=True,
        profile_memory=True,
    )

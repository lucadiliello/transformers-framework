
from deepspeed.runtime.config import get_fp16_enabled
from deepspeed.runtime.config_utils import get_scalar_param
from deepspeed.runtime.constants import (
    FP16,
    FP16_HYSTERESIS,
    FP16_HYSTERESIS_DEFAULT,
    FP16_INITIAL_SCALE_POWER,
    FP16_INITIAL_SCALE_POWER_DEFAULT,
    FP16_LOSS_SCALE_WINDOW,
    FP16_LOSS_SCALE_WINDOW_DEFAULT,
    FP16_MIN_LOSS_SCALE,
    FP16_MIN_LOSS_SCALE_DEFAULT,
)
from deepspeed.runtime.fp16.loss_scaler import DELAYED_SHIFT, INITIAL_LOSS_SCALE, MIN_LOSS_SCALE, SCALE_WINDOW


RAISE_ERROR_AT_MIN_SCALE = "raise_error_at_min_scale"
RAISE_ERROR_AT_MIN_SCALE_DEFAULT = False


def get_dynamic_loss_scale_args(param_dict):
    loss_scale_args = None
    if get_fp16_enabled(param_dict):
        fp16_dict = param_dict[FP16]
        dynamic_loss_args = [
            FP16_INITIAL_SCALE_POWER,
            FP16_LOSS_SCALE_WINDOW,
            FP16_MIN_LOSS_SCALE,
            FP16_HYSTERESIS,
        ]
        if any(arg in list(fp16_dict.keys()) for arg in dynamic_loss_args):
            init_scale = get_scalar_param(fp16_dict, FP16_INITIAL_SCALE_POWER, FP16_INITIAL_SCALE_POWER_DEFAULT)
            scale_window = get_scalar_param(fp16_dict, FP16_LOSS_SCALE_WINDOW, FP16_LOSS_SCALE_WINDOW_DEFAULT)
            delayed_shift = get_scalar_param(fp16_dict, FP16_HYSTERESIS, FP16_HYSTERESIS_DEFAULT)
            min_loss_scale = get_scalar_param(fp16_dict, FP16_MIN_LOSS_SCALE, FP16_MIN_LOSS_SCALE_DEFAULT)
            raise_error = get_scalar_param(fp16_dict, RAISE_ERROR_AT_MIN_SCALE, RAISE_ERROR_AT_MIN_SCALE_DEFAULT)

            loss_scale_args = {
                INITIAL_LOSS_SCALE: 2**init_scale,
                SCALE_WINDOW: scale_window,
                DELAYED_SHIFT: delayed_shift,
                MIN_LOSS_SCALE: min_loss_scale,
                RAISE_ERROR_AT_MIN_SCALE: raise_error,
            }

    return loss_scale_args

import logging
import warnings
from typing import Any, Dict, Iterable, Mapping, Union

import torch
from lightning_fabric.utilities.rank_zero import rank_zero_only


def formatwarning(message, *args, **kwargs):
    return f"{message}"


warnings.formatwarning = formatwarning

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(name)s [%(levelname)s]: %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
# setting main framework logger
logger = logging.getLogger('transformers_framework')
logger.setLevel(logging.INFO)
logger.addHandler(ch)

# manage warning logging to be less verbose
logging.captureWarnings(True)
warning_logging = logging.getLogger('py.warnings')
warning_logging.addHandler(ch)

# manage lightning logging
pl_logger = logging.getLogger('lightning.pytorch')
pl_logger.setLevel(logging.INFO)
pl_logger.addHandler(ch)
pl_logger.propagate = False

pl_logger = logging.getLogger('lightning.pytorch.utilities.rank_zero')
pl_logger.setLevel(logging.INFO)
pl_logger.addHandler(ch)
pl_logger.propagate = False

datasets_logger = logging.getLogger('datasets')
datasets_logger.setLevel(logging.INFO)
datasets_logger.addHandler(ch)

torch_logger = logging.getLogger('torch')
torch_logger.setLevel(logging.INFO)
torch_logger.addHandler(ch)


@rank_zero_only
def rank_zero_debug(*args: Any, **kwargs: Any) -> None:
    r""" Function used to log debug-level messages only on global rank 0. """
    logger.debug(*args, **kwargs)


@rank_zero_only
def rank_zero_info(*args: Any, **kwargs: Any) -> None:
    r""" Function used to log info-level messages only on global rank 0. """
    logger.info(*args, **kwargs)


@rank_zero_only
def rank_zero_warn(message: Union[str, Warning], **kwargs: Any) -> None:
    r""" Function used to log warn-level messages only on global rank 0. """
    logger.warn(message, **kwargs)


@rank_zero_only
def rank_zero_error(*args: Any, **kwargs: Any) -> None:
    r""" Function used to log error-level messages only on global rank 0. """
    logger.error(*args, **kwargs)


def parse_log_arguments(*args) -> Dict[str, Union[int, float, torch.Tensor]]:
    r""" Parse log arguments for LightningModule before logging with original method. """

    if not (1 <= len(args) <= 2):
        raise ValueError(
            "log method should be called with only 1 (Mapping) or two "
            "(name, Union[Iterable, int, float, Tensor]) positional arguments"
        )

    if len(args) == 1:  # user must have passed a dict
        data = args[0]
    else:
        name, value = args
        # start with single built-in or Tensor values
        if (isinstance(value, torch.Tensor) and value.dim() == 0) or isinstance(value, (int, float)):
            data = {name: value}
        elif isinstance(value, Iterable):
            data = {f"{name}_class_{i}": v for i, v in enumerate(value)}
            data[name] = value.mean() if isinstance(value, torch.Tensor) else sum(value) / len(value)

    if not isinstance(data, Mapping):
        raise ValueError("Log value must be an Iterable, a Mapping or a single int/float")

    # check user is not logging metrics but only allowed values
    for v in data.values():
        if not isinstance(v, (int, float, torch.Tensor)):
            raise ValueError(
                "`transformers_framework` automatically manages metrics, just log integers, floats or an "
                "iterable or the previouses. Please use metric(...) and not metric.compute() to avoid overheads"
            )

    return data

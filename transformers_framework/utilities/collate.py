from typing import Any, Dict, List, Union

import numpy as np
import numpy.typing as npt

from transformers_framework.utilities.functional import list2dict


def collate_flexible_numpy_fn(
    data: List[Dict[str, Union[npt.NDArray, Any]]],
) -> Dict[str, Union[npt.NDArray, List[Any]]]:
    r"""
    Stack together numpy arrays with the same key name and just concatenate other values into a list.

    Args:
        data: List of dictionaries with the values to merge

    Returns:
        a dict with the preprocessed batch of numpy arrays
    """

    if not data:
        return dict()

    # check inputs are well formed
    data = list2dict(data)
    assert all(data.values()), "cannot process and empty batch"  # nosec
    assert all(all(type(v) == type(value[0]) for v in value) for value in data.values()), (  # noqa: E721 # nosec
        "all values for each key across all input dictionaries must be of the same type"
    )

    # create output batch stacking together numpy tensors
    batch = {}
    for key, values in data.items():
        if isinstance(values[0], np.ndarray):
            batch[key] = np.stack(values, axis=0)
        elif all(v is None for v in values):
            batch[key] = None
        else:
            batch[key] = values

    return batch

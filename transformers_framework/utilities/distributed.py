from typing import Any, List

import torch


def distributed_available() -> bool:
    return torch.distributed.is_available()


def distributed_initialized() -> bool:
    return torch.distributed.is_initialized()


def sync_data_distributed(obj: Any, group: int = None, concat: bool = False) -> List[Any]:
    r""" Syncronize a pickeable object across the world and return concatenation of
    every object in a list. """

    if concat is True:
        if not isinstance(obj, (tuple, list)):
            raise ValueError(
                f"to concat in `sync_data_distributed` you must pass a list or a tuple, got {type(obj)}"
            )

    if distributed_available():
        world_size = torch.distributed.get_world_size(group=group)
        res = [None] * world_size
        torch.distributed.all_gather_object(res, obj, group=group)

        if concat is True:
            res = [x for r in res for x in r]
        return res
    
    return obj if concat else [obj]  # keep this for compatibility between single device and distributed

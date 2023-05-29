from typing import Callable


def native(method: Callable) -> Callable:
    method._native = True
    return method


def is_overidden(method: Callable) -> bool:
    return not hasattr(method, "_native") or not method._native

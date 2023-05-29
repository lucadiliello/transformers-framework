import hashlib
import inspect
from typing import Callable


def hash_function(func: Callable) -> str:
    r"""
    Produces a hash for the code in the given function.

    Args:
        func (types.FunctionObject): The function to produce a hash for
    """

    func_str = inspect.getsource(func)

    # Produce the hash
    fhash = hashlib.sha256(func_str.encode())
    result = fhash.hexdigest()
    return result


def hash_string(string: str) -> str:
    r""" Produces a hash for the given string. """
    return hashlib.sha1(string.encode("utf-8")).hexdigest()

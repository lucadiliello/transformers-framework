import re
import warnings
from copy import copy


def partial_format(string: str, **kwargs) -> str:
    r""" Substitute only a portion of the keyword placeholders. """
    res = copy(string)
    instances = re.finditer('\{.*?\}', res)  # noqa: W605

    for instance in instances:
        start = instance.start()
        end = instance.end()
        text = res[start:end]

        colon_index = text.find(':')

        # colon_index can be -1, 1 ... len(string) -1
        if colon_index == -1:
            name = text[1:-1]  # discard {} brackets
        else:
            name = text[1:colon_index]

        if name and name in kwargs:
            res = res[:start] + text.format(**{name: kwargs[name]}) + res[end:]
            del kwargs[name]

    if kwargs:
        warnings.warn(f"The following keys were not used to format string: {list(kwargs.keys())}")

    return res

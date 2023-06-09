from typing import Any


class ExtendedConfig:
    r"""
    Adds some useful parameters for extended classification to transformers configurations classes.

    Args:
        k (:obj:`int`, `optional`, defaults to None):
            Number of classifications to perform for each input example.
        classification_head_type (:obj:`str`, `optional`, defaults to None):
            the type of multiple-predictions classification head
    """

    def __init__(self, k: int = None, classification_head_type: str = None, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.classification_head_type = classification_head_type

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == 'head_type':
            raise ValueError("Update `head_type` in your configuration file to `classification_head_type`")
        super().__setattr__(__name, __value)


class MultiTokenConfig:
    r"""
    Configuration with an additional argument regarding the
    maximum number of positional embeddings for the multiple generation.

    Arguments:
        max_multi_token_predictions (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """

    def __init__(self, max_multi_token_predictions: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.max_multi_token_predictions = max_multi_token_predictions

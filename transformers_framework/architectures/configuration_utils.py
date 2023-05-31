from transformers.configuration_utils import logger


class ExtendedConfig(object):
    r"""
    Adds some useful parameters for extended classification to transformers configurations classes.

    Args:
        k (:obj:`int`, `optional`, defaults to None):
            Number of classifications to perform for each input example.
        classification_head_type (:obj:`str`, `optional`, defaults to None):
            the type of multiple-predictions classification head
    """

    def __init__(self, k: int = None, classification_head_type: str = None, head_type: str = None, **kwargs):
        super().__init__()
        self.k = k

        if head_type is not None:
            logger.warning("You passed deprecated argument `head_type`, converting it to `classification_head_type`.")
        classification_head_type = head_type

        assert classification_head_type in ('IE_1', 'AE_1', 'IE_k', 'AE_k', 'RE_k')
        self.classification_head_type = classification_head_type


class MultiTokenConfig(object):
    r"""
    Configuration with an additional argument regarding the
    maximum number of positional embeddings for the multiple generation.

    Arguments:
        max_multi_token_predictions (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """

    def __init__(self, max_multi_token_predictions: int = 12, **kwargs):
        super().__init__()

        if not isinstance(max_multi_token_predictions, int) or max_multi_token_predictions < 1:
            raise ValueError("`max_multi_token_predictions` must be an integer greater than 0")

        self.max_multi_token_predictions = max_multi_token_predictions

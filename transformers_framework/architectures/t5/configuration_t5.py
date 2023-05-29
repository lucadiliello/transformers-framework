from transformers.models.t5.configuration_t5 import T5Config


class T5MultiTokenConfig(T5Config):
    r"""
    Same as T5 Configuration with an additional argument with the
    maximum number of positional embeddings for the multiple generation.

    Arguments:
        max_multi_token_predictions (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """

    def __init__(self, max_multi_token_predictions: int = 12, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(max_multi_token_predictions, int) or max_multi_token_predictions < 1:
            raise ValueError("`max_multi_token_predictions` must be an integer greater than 0")

        self.max_multi_token_predictions = max_multi_token_predictions

from transformers.models.t5.configuration_t5 import T5Config

from transformers_framework.architectures.configuration_utils import MultiTokenConfig


class T5MultiTokenConfig(MultiTokenConfig, T5Config):
    r"""
    Same as T5 Configuration with an additional argument with the
    maximum number of positional embeddings for the multiple generation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

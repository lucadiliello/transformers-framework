from transformers.models.t5.configuration_t5 import T5Config

from transformers_framework.architectures.configuration_utils import MultiTokenConfig


class T5MultiTokenConfig(T5Config, MultiTokenConfig):
    r"""
    Same as T5 Configuration with an additional argument with the
    maximum number of positional embeddings for the multiple generation.
    """

    def __init__(self, **kwargs):
        super(T5Config, self).__init__(**kwargs)
        super(MultiTokenConfig, self).__init__(**kwargs)

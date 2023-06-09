from transformers.models.bart import BartConfig

from transformers_framework.architectures.configuration_utils import MultiTokenConfig


class BartMultiTokenConfig(MultiTokenConfig, BartConfig):
    r"""
    Same as BART Configuration with an additional argument with the
    maximum number of positional embeddings for the multiple generation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

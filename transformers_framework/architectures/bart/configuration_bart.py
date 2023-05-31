from transformers.models.bart import BartConfig

from transformers_framework.architectures.configuration_utils import MultiTokenConfig


class BartMultiTokenConfig(BartConfig, MultiTokenConfig):
    r"""
    Same as BART Configuration with an additional argument with the
    maximum number of positional embeddings for the multiple generation.
    """

    def __init__(self, **kwargs):
        super(BartConfig, self).__init__(**kwargs)
        super(MultiTokenConfig, self).__init__(**kwargs)

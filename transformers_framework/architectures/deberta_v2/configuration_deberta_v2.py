from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config

from transformers_framework.architectures.configuration_utils import ExtendedConfig


class DebertaV2ExtendedConfig(ExtendedConfig, DebertaV2Config):
    r"""
    The :class:`~transformers.DebertaV2Config` reuses the
    same defaults. Please check the parent classes for more information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

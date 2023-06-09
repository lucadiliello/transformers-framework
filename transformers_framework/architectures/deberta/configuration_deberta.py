from transformers.models.deberta.configuration_deberta import DebertaConfig

from transformers_framework.architectures.configuration_utils import ExtendedConfig


class DebertaExtendedConfig(ExtendedConfig, DebertaConfig):
    r"""
    The :class:`~transformers.DebertaConfig` reuses the
    same defaults. Please check the parent classes for more information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

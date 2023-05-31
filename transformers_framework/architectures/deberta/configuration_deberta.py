from transformers.models.deberta.configuration_deberta import DebertaConfig

from transformers_framework.architectures.configuration_utils import ExtendedConfig


class DebertaExtendedConfig(DebertaConfig, ExtendedConfig):
    r"""
    The :class:`~transformers.DebertaConfig` reuses the
    same defaults. Please check the parent classes for more information.
    """

    def __init__(self, **kwargs):
        super(DebertaConfig, self).__init__(**kwargs)
        super(ExtendedConfig, self).__init__(**kwargs)

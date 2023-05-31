from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config

from transformers_framework.architectures.configuration_utils import ExtendedConfig


class DebertaV2ExtendedConfig(DebertaV2Config, ExtendedConfig):
    r"""
    The :class:`~transformers.DebertaV2Config` reuses the
    same defaults. Please check the parent classes for more information.
    """

    def __init__(self, **kwargs):
        super(DebertaV2Config, self).__init__(**kwargs)
        super(ExtendedConfig, self).__init__(**kwargs)

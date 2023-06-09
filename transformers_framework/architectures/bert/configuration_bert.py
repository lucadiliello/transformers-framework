from transformers.models.bert.configuration_bert import BertConfig

from transformers_framework.architectures.configuration_utils import ExtendedConfig


class BertExtendedConfig(ExtendedConfig, BertConfig):
    r"""
    The :class:`~transformers.BertConfig` reuses the
    same defaults. Please check the parent classes for more information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

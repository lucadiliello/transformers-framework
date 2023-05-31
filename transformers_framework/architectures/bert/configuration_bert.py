from transformers.models.bert.configuration_bert import BertConfig

from transformers_framework.architectures.configuration_utils import ExtendedConfig


class BertExtendedConfig(BertConfig, ExtendedConfig):
    r"""
    The :class:`~transformers.BertConfig` reuses the
    same defaults. Please check the parent classes for more information.
    """

    def __init__(self, **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        super(ExtendedConfig, self).__init__(**kwargs)

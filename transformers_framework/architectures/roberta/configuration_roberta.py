from transformers.models.roberta.configuration_roberta import RobertaConfig

from transformers_framework.architectures.configuration_utils import ExtendedConfig


class RobertaExtendedConfig(RobertaConfig, ExtendedConfig):
    r"""
    The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`. It reuses the
    same defaults. Please check the parent classes for more information.
    """

    def __init__(self, **kwargs):
        super(RobertaConfig, self).__init__(**kwargs)
        super(ExtendedConfig, self).__init__(**kwargs)

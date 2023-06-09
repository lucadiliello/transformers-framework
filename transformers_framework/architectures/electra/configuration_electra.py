from transformers.models.electra.configuration_electra import ElectraConfig

from transformers_framework.architectures.configuration_utils import ExtendedConfig


class ElectraExtendedConfig(ExtendedConfig, ElectraConfig):
    r"""
    The :class:`~transformers.ElectraConfig` class directly inherits :class:`~transformers.BertConfig`. It reuses the
    same defaults. Please check the parent classes for more information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

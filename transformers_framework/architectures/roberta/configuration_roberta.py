from transformers.models.roberta.configuration_roberta import RobertaConfig


class RobertaExtendedConfig(RobertaConfig):
    r"""
    The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`. It reuses the
    same defaults. Please check the parent class for more information.

    Args:
        k (:obj:`int`, `optional`, defaults to None):
            Number of classifications to perform for each input example.
        classification_head_type (:obj:`str`, `optional`, defaults to 'IE_1'):
            the type of multiple-predictions classification head
    """

    def __init__(self, k: int = None, classification_head_type: str = 'IE_1', **kwargs):
        super().__init__(**kwargs)
        self.k = k

        assert classification_head_type in ('IE_1', 'AE_1', 'IE_k', 'AE_k', 'RE_k')
        self.classification_head_type = classification_head_type

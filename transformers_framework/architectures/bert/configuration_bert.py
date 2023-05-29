from transformers.models.bert.configuration_bert import BertConfig


class BertExtendedConfig(BertConfig):
    r"""
    The :class:`~transformers.BertConfig` reuses the
    same defaults. Please check the parent class for more information.

    Args:
        k (:obj:`int`, `optional`, defaults to None):
            Number of classifications to perform for each input example.
        aggregate_hidden_states (:obj:`bool`, `optional`, defaults to False):
            Aggregate all hidden states to compute classification logits.
    """

    def __init__(self, k: int = None, aggregate_hidden_states: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.aggregate_hidden_states = aggregate_hidden_states

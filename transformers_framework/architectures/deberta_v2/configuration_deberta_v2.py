from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config


class DebertaV2ExtendedConfig(DebertaV2Config):
    r"""
    The :class:`~transformers.DebertaV2Config` reuses the
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

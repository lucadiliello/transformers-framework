from torch import Tensor
from torchmetrics.text.perplexity import Perplexity as _Perplexity

from transformers_framework.utilities.metrics import perplexity_update


class Perplexity(_Perplexity):
    r""" Perplexity measures how well a language model predicts a text sample. It's calculated as the average number
    of bits per word a model needs to represent the sample.

    Args:
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score.
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Examples:
        >>> import torch
        >>> preds = torch.rand(2, 8, 5, generator=torch.manual_seed(22))
        >>> target = torch.randint(5, (2, 8), generator=torch.manual_seed(22))
        >>> target[0, 6:] = -100
        >>> metric = Perplexity(ignore_index=-100)
        >>> metric(preds, target)
        tensor(5.2545)
    """

    def update(self, preds: Tensor, target: Tensor) -> None:
        r""" Compute and store intermediate statistics for Perplexity.

        Args:
            preds:
                Probabilities assigned to each token in a sequence with shape [batch_size, seq_len, vocab_size].
            target:
                Ground truth values with a shape [batch_size, seq_len].
        """
        total_log_probs, count = perplexity_update(preds, target, self.ignore_index)
        self.total_log_probs += total_log_probs
        self.count += count

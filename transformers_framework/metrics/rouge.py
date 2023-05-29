from typing import Any, Callable, Dict, Literal, Sequence, Tuple, Union

from torch import Tensor
from torchmetrics.text.rouge import ROUGEScore as _ROUGEScore


class ROUGEScore(_ROUGEScore):
    r"""`Calculate Rouge Score`_, used for automatic summarization.

    This implementation should imitate the behaviour of the `rouge-score` package `Python ROUGE Implementation`

    Args:
        use_stemmer: Use Porter stemmer to strip word suffixes to improve matching.
        normalizer: A user's own normalizer function.
            If this is ``None``, replacing any non-alpha-numeric characters with spaces is default.
            This function must take a ``str`` and return a ``str``.
        tokenizer:
            A user's own tokenizer function. If this is ``None``, spliting by spaces is default
            This function must take a `str` and return ``Sequence[str]``
        accumulate:
            Useful in case of multi-reference rouge score.

            - ``avg`` takes the avg of all references with respect to predictions
            - ``best`` takes the best fmeasure score obtained between prediction and multiple corresponding references.

        rouge_keys: A list of rouge types to calculate.
            Keys that are allowed are ``rougeL``, ``rougeLsum``, and ``rouge1`` through ``rouge9``.
        name: A string that will be prepended to all metrics before returning the logging dict.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.text.rouge import ROUGEScore
        >>> preds = "My name is John"
        >>> target = "Is your name John"
        >>> rouge = ROUGEScore(name="summarization")
        >>> from pprint import pprint
        >>> pprint(rouge(preds, target))
        {'summarization/rouge1_fmeasure': tensor(0.7500),
         'summarization/rouge1_precision': tensor(0.7500),
         'summarization/rouge1_recall': tensor(0.7500),
         'summarization/rouge2_fmeasure': tensor(0.),
         'summarization/rouge2_precision': tensor(0.),
         'summarization/rouge2_recall': tensor(0.),
         'summarization/rougeL_fmeasure': tensor(0.5000),
         'summarization/rougeL_precision': tensor(0.5000),
         'summarization/rougeL_recall': tensor(0.5000),
         'summarization/rougeLsum_fmeasure': tensor(0.5000),
         'summarization/rougeLsum_precision': tensor(0.5000),
         'summarization/rougeLsum_recall': tensor(0.5000)}


    Raises:
        ValueError:
            If the python packages ``nltk`` is not installed.
        ValueError:
            If any of the ``rouge_keys`` does not belong to the allowed set of keys.

    References:
        [1] ROUGE: A Package for Automatic Evaluation of Summaries by Chin-Yew Lin `Rouge Detail`_
    """
    def __init__(
        self,
        use_stemmer: bool = False,
        normalizer: Callable[[str], str] = None,
        tokenizer: Callable[[str], Sequence[str]] = None,
        accumulate: Literal["avg", "best"] = "best",
        rouge_keys: Union[str, Tuple[str, ...]] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),
        name: str = None,
        **kwargs: Any,
    ):
        super().__init__(
            use_stemmer=use_stemmer,
            normalizer=normalizer,
            tokenizer=tokenizer,
            accumulate=accumulate,
            rouge_keys=rouge_keys,
            **kwargs,
        )
        self.name = name

    def compute(self) -> Dict[str, Tensor]:
        r""" Calculate (Aggregate and provide confidence intervals) ROUGE score.

        Return:
            Python dictionary of rouge scores for each input rouge key.
        """
        res = super().compute()
        if self.name is not None:
            res = {f"{self.name}/{k}": v for k, v in res.items()}

        return res

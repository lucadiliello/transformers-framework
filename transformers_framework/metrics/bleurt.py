from typing import Any, Dict, List, Optional, Union
from warnings import warn

import torch
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torchmetrics.metric import Metric

from transformers_framework.utilities.tokenization import advanced_tokenization
from transformers_framework.utilities.torch import split_batch


_BLEURT_PYTORCH_AVAILABLE = RequirementCache("bleurt_pytorch")
if _BLEURT_PYTORCH_AVAILABLE:
    from bleurt_pytorch.bleurt.configuration_bleurt import BleurtConfig
    from bleurt_pytorch.bleurt.modeling_bleurt import BleurtForSequenceClassification
    from bleurt_pytorch.bleurt.tokenization_bleurt import BleurtTokenizer
else:
    BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer = object, object, object


# Default model recommended in the original implementation.
_DEFAULT_MODEL = "lucadiliello/BLEURT-20"


class BLEURT(Metric):
    r""" Google's `BLUERT`_ leverages the pre-trained contextual embeddings from BERT and
    matches words in candidate and reference sentences by cross-embedding. It has been shown to correlate with
    human judgment on sentence-level and system-level evaluation.

    This implemenation is a custom implementation and follows the original implementation from `BERT_score`_.

    Args:
        model_name_or_path: A name or a model path used to load `transformers` pretrained model.
        max_length: A maximum length of input sequences. Sequences longer than `max_length` are to be trimmed.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        Single tensor containing BLEURT similarity.

    Example:
        >>> from transformers_framework.metrics.bleurt import BLEURT
        >>> preds = ["hello there", "general kenobi"]
        >>> target = ["hello there", "master kenobi"]
        >>> bleurt = BLEURT('lucadiliello/BLEURT-20-D6')
        >>> bleurt.load()
        >>> score = bleurt(preds, target)
        >>> bleurt.unload()
        >>> from pprint import pprint
        tensor(0.8310, grad_fn=<SqueezeBackward0>)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    preds_input_ids: List[Tensor]
    preds_attention_mask: List[Tensor]
    preds_token_type_ids: List[Tensor]
    target_input_ids: List[Tensor]
    target_attention_mask: List[Tensor]
    target_token_type_ids: List[Tensor]

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        max_length: int = None,
        aggregate_fn: str = 'max',
        **kwargs: Any
    ):
        super().__init__(**kwargs)

        assert _BLEURT_PYTORCH_AVAILABLE, (
            "BLEURT metric needs `bleurt-pytorch` to be installed. "
            "Please run `pip install git+https://github.com/lucadiliello/bleurt-pytorch.git`"
        )

        if model_name_or_path is None:
            warn(
                f"The argument `model_name_or_path` was not specified while it is required when the default "
                f"`transformers` model is used. It will use the default recommended model - {_DEFAULT_MODEL!r}."
            )

        self.model_name_or_path = model_name_or_path or _DEFAULT_MODEL

        self.config = BleurtConfig.from_pretrained(self.model_name_or_path)
        self.tokenizer = BleurtTokenizer.from_pretrained(self.model_name_or_path)

        if max_length is not None:
            self.max_length = max_length
        elif hasattr(self.tokenizer, "max_model_input_sizes") and (
            self.model_name_or_path in self.tokenizer.max_model_input_sizes
        ):
            self.max_length = self.tokenizer.max_model_input_sizes[self.model_name_or_path]
        elif hasattr(self.tokenizer, "max_len_single_sentence"):
            self.max_length = self.tokenizer.max_len_single_sentence + self.tokenizer.num_special_tokens_to_add(
                pair=False
            )
        else:
            self.max_length = self.config.max_position_embeddings

        self.batch_size = None

        # aggregation fn to torch fn
        if aggregate_fn not in ('mean', 'max', 'min'):
            raise ValueError(f"`{aggregate_fn}` must be one of `mean`, `max` and `min`")
        self.aggregate_fn = aggregate_fn

        self.add_state("parts_length", [], dist_reduce_fx="cat")
        self.add_state("input_ids", [], dist_reduce_fx="cat")
        self.add_state("attention_mask", [], dist_reduce_fx="cat")
        self.add_state("token_type_ids", [], dist_reduce_fx="cat")

    def load(self):
        r""" Create and load model in memory. """
        self.model = BleurtForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.model.eval()
        self.model.to(self.device)

    def unload(self):
        r""" Delete model and remove from memory. """
        del self.model

    def update(self, preds: List[str], target: Union[List[str], List[List[str]]]):
        r""" Store predictions/references for computing BLEURT scores. It is necessary to store sentences in a
        tokenized form to ensure the DDP mode working.

        Args:
            preds: An iterable of predicted sentences.
            target: An iterable of reference sentences or a nested iterable with a set of targets for each pred.
        """
        if not len(preds) or len(preds) != len(target):
            raise ValueError("`preds` and `target` must be two empty lists with equal length")

        if len(preds) != self.batch_size:
            self.batch_size = len(preds)

        # force nested list for targets
        for i in range(len(target)):
            if isinstance(target[i], str):
                target[i] = [target[i]]

        # keep track of number of targets for each input prediction
        parts_length = torch.tensor([len(t) for t in target], device=self.device, dtype=torch.int64)

        input_dicts = [
            advanced_tokenization(
                [p] * len(t), t, tokenizer=self.tokenizer, max_sequence_length=self.max_length, squeeze=False
            )
            for p, t in zip(preds, target)
        ]

        # convert dict of lists to dict of tensors
        input_dicts = [
            {k: torch.tensor(v) for k, v in input_dict.items()} for input_dict in input_dicts
        ]

        # update internal states
        self.parts_length.append(parts_length)
        for input_dict in input_dicts:
            self.input_ids.append(input_dict["input_ids"])
            self.attention_mask.append(input_dict["attention_mask"])
            self.token_type_ids.append(input_dict["token_type_ids"])

    def compute(self) -> Dict[str, float]:
        r""" Calculate BERT scores.

        Return:
            Python dictionary containing the keys `bert_score_precision`, `bert_score_recall` and `bert_score_f1`
            with corresponding values.
        """

        input_dict = dict(
            input_ids=torch.cat(self.input_ids).to(device=self.device),
            attention_mask=torch.cat(self.attention_mask).to(device=self.device),
            token_type_ids=torch.cat(self.token_type_ids).to(device=self.device),
        )
        parts_length = torch.cat(self.parts_length)

        # batched processing
        res = []
        with torch.inference_mode():
            for input_dict in split_batch(input_dict, self.batch_size):
                res.append(self.model(**input_dict).logits.cpu())

        # rebuild results
        scores = torch.cat(res, dim=0)

        # parse torch aggregation fn
        if self.aggregate_fn == 'mean':
            aggregate_fn = torch.mean
        elif self.aggregate_fn == 'max':
            aggregate_fn = torch.max
        elif self.aggregate_fn == 'min':
            aggregate_fn = torch.min

        # compute micro average and then average over all samples
        scores = [aggregate_fn(score) for score in torch.split(scores, parts_length.detach().cpu().tolist())]
        res = torch.stack(scores).mean()

        return res

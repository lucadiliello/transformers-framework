from typing import Any, Dict, List, Optional, Union
from warnings import warn

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from transformers import AutoConfig, AutoModel, AutoTokenizer

from transformers_framework.utilities.metrics import bert_score
from transformers_framework.utilities.tokenization import advanced_tokenization
from transformers_framework.utilities.torch import split_batch


# Default model recommended in the original implementation.
_DEFAULT_MODEL = "roberta-large"


class BERTScore(Metric):
    r""" `Bert_score Evaluating Text Generation`_ leverages the pre-trained contextual embeddings from BERT and
    matches words in candidate and reference sentences by cosine similarity. It has been shown to correlate with
    human judgment on sentence-level and system-level evaluation. Moreover, BERTScore computes precision, recall,
    and F1 measure, which can be useful for evaluating different language generation tasks.

    This implemenation is a custom implementation and follows the original implementation from `BERT_score`_.

    Args:
        model_name_or_path: A name or a model path used to load `transformers` pretrained model.
        max_length: A maximum length of input sequences. Sequences longer than `max_length` are to be trimmed.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        Python dictionary containing the keys `precision`, `recall` and `f1` with corresponding values.

    Example:
        >>> from transformers_framework.metrics.bertscore import BERTScore
        >>> preds = ["hello there", "general kenobi"]
        >>> target = ["hello there", "master kenobi"]
        >>> bertscore = BERTScore()
        >>> bertscore.load()
        >>> score = bertscore(preds, target)
        >>> bertscore.unload()
        >>> from pprint import pprint
        >>> rounded_score = {k: [round(v, 3) for v in vv] for k, vv in score.items()}
        >>> pprint(rounded_score)
        {'f1': 0.998, 'precision': 0.998, 'recall': 0.998}
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    preds_input_ids: List[Tensor]
    preds_attention_mask: List[Tensor]
    target_input_ids: List[Tensor]
    target_attention_mask: List[Tensor]

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        max_length: int = None,
        name: str = 'bert_score',
        aggregate_fn: str = 'max',
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if model_name_or_path is None:
            warn(
                f"The argument `model_name_or_path` was not specified while it is required when the default "
                f"`transformers` model is used. It will use the default recommended model - {_DEFAULT_MODEL!r}."
            )

        self.model_name_or_path = model_name_or_path or _DEFAULT_MODEL

        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

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

        # remember batch size
        self.batch_size = None

        # aggregation fn to torch fn
        if aggregate_fn not in ('mean', 'max', 'min'):
            raise ValueError(f"`{aggregate_fn}` must be one of `mean`, `max` and `min`")
        self.aggregate_fn = aggregate_fn

        if not isinstance(name, str):
            raise ValueError("Name cannot be different from string.")
        self.name = name

        self.add_state("parts_length", [], dist_reduce_fx="cat")
        self.add_state("preds_input_ids", [], dist_reduce_fx="cat")
        self.add_state("preds_attention_mask", [], dist_reduce_fx="cat")
        self.add_state("target_input_ids", [], dist_reduce_fx="cat")
        self.add_state("target_attention_mask", [], dist_reduce_fx="cat")

    def load(self):
        r""" Create and load model in memory. """
        self.model = AutoModel.from_pretrained(self.model_name_or_path)
        self.model.eval()
        self.model.to(self.device)

    def unload(self):
        r""" Delete model and remove from memory. """
        del self.model

    def update(self, preds: List[str], target: Union[List[str], List[List[str]]]):
        r""" Store predictions/references for computing BERT scores. It is necessary to store sentences in a
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

        # compute input ids and attention masks
        preds_dicts = [
            advanced_tokenization(
                [p] * len(t), tokenizer=self.tokenizer, max_sequence_length=self.max_length, squeeze=False
            )
            for p, t in zip(preds, target)
        ]
        target_dicts = [
            advanced_tokenization(t, tokenizer=self.tokenizer, max_sequence_length=self.max_length, squeeze=False)
            for t in target
        ]

        # convert dict of lists to dict of tensors
        preds_dicts = [
            {k: torch.tensor(v) for k, v in preds_dict.items()} for preds_dict in preds_dicts
        ]
        target_dicts = [
            {k: torch.tensor(v) for k, v in target_dict.items()} for target_dict in target_dicts
        ]

        # update internal states
        self.parts_length.append(parts_length)
        for preds_dict, target_dict in zip(preds_dicts, target_dicts):
            self.preds_input_ids.append(preds_dict["input_ids"])
            self.preds_attention_mask.append(preds_dict["attention_mask"])
            self.target_input_ids.append(target_dict["input_ids"])
            self.target_attention_mask.append(target_dict["attention_mask"])

    def compute(self) -> Dict[str, float]:
        r""" Calculate BERT scores.

        Return:
            Python dictionary containing the keys `bert_score_precision`, `bert_score_recall` and `bert_score_f1`
            with corresponding values.
        """

        preds = dict(input_ids=torch.cat(self.preds_input_ids), attention_mask=torch.cat(self.preds_attention_mask))
        target = dict(input_ids=torch.cat(self.target_input_ids), attention_mask=torch.cat(self.target_attention_mask))
        parts_length = torch.cat(self.parts_length).detach().cpu().tolist()

        # batched processing
        res = []
        with torch.inference_mode():
            for preds_batch, target_batch in zip(
                split_batch(preds, self.batch_size),
                split_batch(target, self.batch_size)
            ):
                res.append(
                    bert_score(
                        preds=preds_batch,
                        target=target_batch,
                        model=self.model,
                    )
                )

        # rebuild results
        precision = torch.cat([r[0] for r in res], dim=0)
        recall = torch.cat([r[1] for r in res], dim=0)
        f1_score = torch.cat([r[2] for r in res], dim=0)

        # parse torch aggregation fn
        if self.aggregate_fn == 'mean':
            aggregate_fn = torch.mean
        elif self.aggregate_fn == 'max':
            aggregate_fn = torch.max
        elif self.aggregate_fn == 'min':
            aggregate_fn = torch.min

        # compute micro average and then average over all samples
        precision = [aggregate_fn(p) for p in torch.split(precision, parts_length, dim=0)]
        recall = [aggregate_fn(r) for r in torch.split(recall, parts_length, dim=0)]
        f1_score = [aggregate_fn(f) for f in torch.split(f1_score, parts_length, dim=0)]

        precision = torch.stack(precision).mean()
        recall = torch.stack(recall).mean()
        f1_score = torch.stack(f1_score).mean()

        return {
            f"{self.name}_f1": precision,
            f"{self.name}_precision": precision,
            f"{self.name}_recall": recall,
        }

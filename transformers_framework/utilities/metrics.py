from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.functional.text.bert import _get_precision_recall_f1
from torchmetrics.functional.text.helper_embedding_metric import _process_attention_mask_for_special_tokens
from torchmetrics.functional.text.perplexity import _check_shape_and_type_consistency
from transformers.modeling_utils import PreTrainedModel


def compute_embeddings(
    batch: Dict,
    model: Module,
) -> Tuple[Tensor, Tensor]:
    r""" Calculate sentence embeddings and the inverse-document-frequency scaling factor.

    Args:
        dataloader: dataloader instance.
        model: BERT model.
        device: A device to be used for calculation.
        name: a name to be shown on progress bar

    Return:
        A tuple of ``torch.Tensor``s containing the model's embeddings and the normalized tokens IDF.
        When ``idf = False``, tokens IDF is not calculated, and a matrix of mean weights is returned instead.
        For a single sentence, ``mean_weight = 1/seq_len``, where ``seq_len`` is a sum over the corresponding
        ``attention_mask``.
    """

    with torch.no_grad():

        # move data to correct device
        batch = {k: v.to(device=model.device) for k, v in batch.items()}

        # Output shape: bs x sequence_length x bert_dim
        model_output = model(**batch, output_hidden_states=True)
        out = (
            model_output.last_hidden_states
            if hasattr(model_output, "last_hidden_states")
            else model_output.hidden_states[-1]
        )
        out = out.unsqueeze(1)  # add num layers dimension, required by torchmetrics functions

        out /= out.norm(dim=-1).unsqueeze(-1)  # normalize embeddings
        processed_attention_mask = _process_attention_mask_for_special_tokens(batch["attention_mask"])

        # Multiply embeddings with attention_mask (b=batch_size, l=num_layers, s=seq_len, d=emb_dim)
        out = torch.einsum("blsd, bs -> blsd", out, processed_attention_mask)
        embeddings = out.cpu()

        # Calculate weighted (w.r.t. sentence length) input_ids IDF matrix
        input_ids_idf = processed_attention_mask.type(out.dtype)
        input_ids_idf /= input_ids_idf.sum(-1, keepdim=True)
        idf_scale = input_ids_idf.cpu()

    return embeddings, idf_scale


def bert_score(
    preds: Dict[str, Tensor],
    target: Dict[str, Tensor],
    model: PreTrainedModel,
) -> Tuple[torch.Tensor]:
    r""" `Bert_score Evaluating Text Generation`_ leverages the pre-trained contextual embeddings from BERT and
    matches words in candidate and reference sentences by cosine similarity.

    It has been shown to correlate with human judgment on sentence-level and system-level evaluation.
    Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for evaluating different
    language generation tasks.

    This implemenation follows the original implementation from `BERT_score`_.

    Args:
        preds: A ``Dict[input_ids, attention_mask]``.
        target: A  ``Dict[input_ids, attention_mask]``.
        model: A ``transformers`` pretrained model.

    Returns:
        Python dictionary containing the keys ``precision``, ``recall`` and ``f1`` with corresponding values.

    Raises:
        ValueError:
            If ``len(preds) != len(target)``.
        ValueError:
            If invalid input is provided.

    Example:
        >>> from torchmetrics.functional.text.bert import bert_score
        >>> preds = ["hello there", "general kenobi"]
        >>> target = ["hello there", "master kenobi"]
        >>> score = bert_score(preds, target)
        >>> from pprint import pprint
        >>> rounded_score = {k: [round(v, 3) for v in vv] for k, vv in score.items()}
        >>> pprint(rounded_score)
        {'f1': 0.998, 'precision': 0.998, 'recall': 0.998}
    """

    if preds.keys() != target.keys():
        raise ValueError("Got different keys in preds and target batch")

    for k in preds:
        if preds[k].shape != target[k].shape:
            raise ValueError(
                f"Number of predicted and reference sentences must be the same, "
                f"got {preds.shape} and {target.shape} instead"
            )

    preds_embeddings, preds_idf_scale = compute_embeddings(preds, model)
    target_embeddings, target_idf_scale = compute_embeddings(target, model)

    precision, recall, f1_score = _get_precision_recall_f1(
        preds_embeddings.float(), target_embeddings.float(), preds_idf_scale, target_idf_scale
    )

    return precision, recall, f1_score


def perplexity_update(preds: Tensor, target: Tensor, ignore_index: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Compute intermediate statistics for Perplexity.

    Args:
        preds:
            Probabilities assigned to each token in a sequence with shape [batch_size, seq_len, vocab_size].
        target:
            Ground truth values with a shape [batch_size, seq_len].
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score.

    Returns:
        Log probabilities, summed over all samples
        Number of samples
    """
    _check_shape_and_type_consistency(preds, target)

    preds = preds.view(-1, preds.shape[-1])
    target = target.view(-1)

    if ignore_index is not None:
        mask = target != ignore_index
        target = target[mask]
        preds = preds[mask]

    probs = torch.softmax(preds, dim=-1)

    probs = torch.gather(probs, 1, target.unsqueeze(-1))
    total_log_probs = -probs.log().sum()
    count = target.numel()

    return total_log_probs, count

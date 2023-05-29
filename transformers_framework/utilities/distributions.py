from typing import Literal

import torch


def sample_from_distribution(logits: torch.Tensor, sample_function: Literal['gumbel', 'multinomial'] = 'gumbel'):
    r"""
    Sample from generator logits either using gumbel distrib or multinomial distribution.
    Reimplement gumbel softmax because there is a bug in torch.nn.functional.gumbel_softmax
    when fp16 is used (https://github.com/pytorch/pytorch/issues/41663).
    Code taken from
    https://github.com/richarddwang/electra_pytorch/blob/9b2533e62cd1b6126feca323fb7b48480b8c2df0/pretrain.py#L318.
    Gumbel softmax is equal to what official ELECTRA code do,
    standard gumbel dist. = -ln(-ln(standard uniform dist.))
    """

    if sample_function == 'gumbel':
        loc = torch.tensor(0., device=logits.device, dtype=logits.dtype)
        scale = torch.tensor(1., device=logits.device, dtype=logits.dtype)
        gumbel_dist = torch.distributions.gumbel.Gumbel(loc, scale)
        res = (logits + gumbel_dist.sample(logits.shape)).argmax(dim=-1)

    elif sample_function == 'multinomial':
        res = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze()

    else:
        raise ValueError("`sample_function` not valid, choose between 'gumbel' and 'multinomial'")

    return res


def expand_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    logits = torch.stack([1 - probs, probs], dim=dim).log()
    return logits


def expand_probabilities(probabilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.stack([1 - probabilities, probabilities], dim=dim)

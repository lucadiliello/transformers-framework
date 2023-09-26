from typing import List

import torch
from sentence_transformers.util import cos_sim, pairwise_dot_score


def cosine_similarity(*embeddings: List[torch.Tensor]) -> torch.Tensor:
    r""" Return the cosine similarity between each pair of embeddings. """
    assert len(embeddings) == 2
    return cos_sim(embeddings[0], embeddings[1])


def dot_product_similarity(*embeddings: List[torch.Tensor]) -> torch.Tensor:
    r""" Return the dot product similarity between each pair of embeddings. """
    assert len(embeddings) == 2
    return pairwise_dot_score(embeddings[0], embeddings[1])

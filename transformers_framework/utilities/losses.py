from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction
from sentence_transformers.losses.BatchSemiHardTripletLoss import BatchSemiHardTripletLoss
from sentence_transformers.losses.ContrastiveLoss import SiameseDistanceMetric
from sentence_transformers.losses.TripletLoss import TripletDistanceMetric
from sentence_transformers.util import cos_sim, pairwise_dot_score
from torch import nn


def cosine_similarity_loss(
    *embeddings: List[torch.Tensor], labels: torch.Tensor = None
) -> torch.Tensor:
    r"""
    Mostly copied from SentenceTransformers.

    Compute cosine similarity between two batches of embeddings.

    Args:
        embeddings (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, hidden_size)`):
            The two batches of embeddings.
        labels (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,)`), default None:
            The labels for each pair of embeddings.

    Return:
       the cosine similarity loss
    """

    assert len(embeddings) == 2
    assert labels is not None

    similarity = torch.cosine_similarity(*embeddings, dim=-1)
    return nn.functional.mse_loss(similarity, labels), similarity


def batch_all_triplet_loss(
    *embeddings: List[torch.Tensor],
    labels: torch.Tensor = None,
    distance_metric: Callable = BatchHardTripletLossDistanceFunction.eucledian_distance,
    margin: float = 5.0,
) -> torch.Tensor:
    r"""
    Mostly copied from SentenceTransformers.

    batch_all_triplet_loss takes a batch with (embeddings, labels) pairs and computes the loss for all possible, valid
    triplets, i.e., anchor and positive must have the same label, anchor and negative a different label. The labels
    must be integers, with same label indicating sentences from the same class. You train dataset
    must contain at least 2 examples per label class.

    | Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    | Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    | Blog post: https://omoindrot.github.io/triplet-loss

    Args:
        embeddings (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, hidden_size)`):
            The batch of embeddings.
        labels (:obj:`torch.IntTensor` of shape :obj:`(batch_size,)`), default None:
            The label which indicate the class of each embedding.
        distance_metric (:obj:`Callable`):
            Function to measure embeddings distance.
        margin (:obj:`float`):
            Minimum margin of the distance.
    
    Return:
        the batchwise triplet loss
    """

    assert len(embeddings) == 1
    assert labels is not None

    pairwise_dist = distance_metric(*embeddings)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = BatchHardTripletLoss.get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss[triplet_loss < 0] = 0

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)

    # num_valid_triplets = mask.sum()
    # fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss


def batch_hard_triplet_loss(
    *embeddings: List[torch.Tensor],
    labels: torch.Tensor = None,
    distance_metric: Callable = BatchHardTripletLossDistanceFunction.eucledian_distance,
    margin: float = 5.0,
) -> torch.Tensor:
    r"""
    Mostly copied from SentenceTransformers.

    batch_hard_triplet_loss takes a batch with (embeddings, labels) pairs and computes the loss for all possible, valid
    triplets, i.e., anchor and positive must have the same label, anchor and negative a different label. It then looks
    for the hardest positive and the hardest negatives.
    The labels must be integers, with same label indicating sentences from the same class. You train dataset
    must contain at least 2 examples per label class. The margin is computed automatically.

    Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    Blog post: https://omoindrot.github.io/triplet-loss

    Args:
        embeddings (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, hidden_size)`):
            The batch of embeddings.
        labels (:obj:`torch.IntTensor` of shape :obj:`(batch_size,)`), default None:
            The label which indicate the class of each embedding.
        distance_metric (:obj:`Callable`):
            Function to measure embeddings distance.
        margin (:obj:`float`):
            Minimum margin of the distance.

    Return:
        the batchwise triplet loss
    """

    assert len(embeddings) == 1
    assert labels is not None

    # Build the triplet loss over a batch of embeddings.
    # For each anchor, we get the hardest positive and hardest negative to form a triplet.

    # Get the pairwise distance matrix
    pairwise_dist = distance_metric(*embeddings)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl[tl < 0] = 0
    triplet_loss = tl.mean()

    return triplet_loss


def batch_hard_soft_margin_triplet_loss(
    *embeddings: List[torch.Tensor],
    labels: torch.Tensor = None,
    distance_metric: Callable = BatchHardTripletLossDistanceFunction.eucledian_distance,
) -> torch.Tensor:
    r"""
    Mostly copied from SentenceTransformers.

    batch_hard_soft_margin_triplet_loss takes a batch with (embeddings, labels) pairs and computes the loss for all
    possible, valid triplets, i.e., anchor and positive must have the same label, anchor and negative a different label.
    The labels must be integers, with same label indicating sentences from the same class. You train dataset
    must contain at least 2 examples per label class. The margin is computed automatically.

    Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    Blog post: https://omoindrot.github.io/triplet-loss

    Args:
        embeddings (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, hidden_size)`):
            The batch of embeddings.
        labels (:obj:`torch.IntTensor` of shape :obj:`(batch_size,)`), default None:
            The label which indicate the class of each embedding.
        distance_metric (:obj:`Callable`):
            Function to measure embeddings distance.

    Return:
        the batchwise triplet loss
    """

    assert len(embeddings) == 1
    assert labels is not None

    # Build the triplet loss over a batch of embeddings.
    # For each anchor, we get the hardest positive and hardest negative to form a triplet.

    # Get the pairwise distance matrix
    pairwise_dist = distance_metric(*embeddings)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss with soft margin
    # tl = hardest_positive_dist - hardest_negative_dist + margin
    # tl[tl < 0] = 0
    tl = torch.log1p(torch.exp(hardest_positive_dist - hardest_negative_dist))

    return tl.mean()


def batch_semi_hard_triplet_loss(
    *embeddings: List[torch.Tensor],
    labels: torch.Tensor = None,
    distance_metric: Callable = BatchHardTripletLossDistanceFunction.eucledian_distance,
    margin: float = 5.0,
) -> torch.Tensor:
    r"""
    Mostly copied from SentenceTransformers.

    batch_semi_hard_triplet_loss takes a batch with (embeddings, labels) pairs and computes the loss for all possible,
    valid triplets, i.e., anchor and positive must have the same label, anchor and negative a different label.
    It then looks for the semi hard positives and negatives.
    The labels must be integers, with same label indicating sentences from the same class. You train dataset
    must contain at least 2 examples per label class. The margin is computed automatically.

    Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    Blog post: https://omoindrot.github.io/triplet-loss

    Args:
        embeddings (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, hidden_size)`):
            The batch of embeddings.
        labels (:obj:`torch.IntTensor` of shape :obj:`(batch_size,)`), default None:
            The label which indicate the class of each embedding.
        distance_metric (:obj:`Callable`):
            Function to measure embeddings distance.
        margin (:obj:`float`):
            Minimum margin of the distance.

    Return:
        the batchwise triplet loss
    """

    assert len(embeddings) == 1
    assert labels is not None

    # Build the triplet loss over a batch of embeddings.
    # We generate all the valid triplets and average the loss over the positive ones.
    labels = labels.unsqueeze(1)

    pdist_matrix = distance_metric(*embeddings)

    adjacency = labels == labels.t()
    adjacency_not = ~adjacency

    batch_size = torch.numel(labels)
    pdist_matrix_tile = pdist_matrix.repeat([batch_size, 1])

    mask = adjacency_not.repeat([batch_size, 1]) & (pdist_matrix_tile > torch.reshape(pdist_matrix.t(), [-1, 1]))

    mask_final = torch.reshape(torch.sum(mask, 1, keepdims=True) > 0.0, [batch_size, batch_size])
    mask_final = mask_final.t()

    negatives_outside = torch.reshape(
        BatchSemiHardTripletLoss._masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]
    )
    negatives_outside = negatives_outside.t()

    negatives_inside = BatchSemiHardTripletLoss._masked_maximum(pdist_matrix, adjacency_not)
    negatives_inside = negatives_inside.repeat([1, batch_size])

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = (pdist_matrix - semi_hard_negatives) + margin

    mask_positives = adjacency.float().to(labels.device) - torch.eye(batch_size, device=labels.device)
    mask_positives = mask_positives.to(labels.device)
    num_positives = torch.sum(mask_positives)

    triplet_loss = torch.sum(
        torch.max(loss_mat * mask_positives, torch.tensor([0.0], device=labels.device))
    ) / num_positives

    return triplet_loss


def contrastive_loss(
    *embeddings: List[torch.Tensor],
    labels: torch.Tensor = None,
    distance_metric: Callable = SiameseDistanceMetric.COSINE_DISTANCE,
    margin: float = 0.5,
) -> torch.Tensor:
    r"""
    Mostly copied from SentenceTransformers.

    Contrastive loss. Expects as input two batches of embeddings and a label of either 0 or 1.
    If the label == 1, then the distance between the two embeddings is reduced.
    If the label == 0, then the distance between the embeddings is increased.

    Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Args:
        embeddings (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, hidden_size)`):
            The two batches of embeddings (anchors, others).
        labels (:obj:`torch.IntTensor` of shape :obj:`(batch_size,)`), default None:
            The labels for each pair of embeddings.
        distance_metric: (:obj:`Callable`):
            Function that returns a distance between two emeddings.
        margin: (:obj:`float`):
            Negative samples (label == 0) should have a distance of at least the margin value.

    Return:
        the contrastive loss
    """

    assert len(embeddings) == 2
    assert labels is not None

    distances = distance_metric(*embeddings)
    losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(margin - distances).pow(2))
    return losses.mean()


def online_contrastive_loss(
    *embeddings: List[torch.Tensor],
    labels: torch.Tensor = None,
    distance_metric: Callable = SiameseDistanceMetric.COSINE_DISTANCE,
    margin: float = 0.5,
) -> torch.Tensor:
    r"""
    Mostly copied from SentenceTransformers.

    Online Contrastive loss. Similar to contrastive_loss, but it selects hard positive (positives that are far apart)
    and hard negative pairs (negatives that are close) and computes the loss only for these pairs. Often yields
    better performances than contrastive_loss.

    Args:
        embeddings (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, hidden_size)`):
            The two batches of embeddings (anchors, others).
        labels (:obj:`torch.IntTensor` of shape :obj:`(batch_size,)`), default None:
            The labels for each pair of embeddings.
        distance_metric: (:obj:`Callable`):
            Function that returns a distance between two emeddings.
        margin: (:obj:`float`):
            Negative samples (label == 0) should have a distance of at least the margin value.

    Return:
        the online contrastive loss
    """

    assert len(embeddings) == 2
    assert labels is not None

    distance_matrix = distance_metric(*embeddings)
    negs = distance_matrix[labels == 0]
    poss = distance_matrix[labels == 1]

    # select hard positive and hard negative pairs
    negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
    positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

    positive_loss = positive_pairs.pow(2).sum()
    negative_loss = F.relu(margin - negative_pairs).pow(2).sum()
    loss = positive_loss + negative_loss
    return loss


def margin_mse_loss(
    *embeddings: List[torch.Tensor],
    labels: torch.Tensor = None,
    similarity_fct: Callable = pairwise_dot_score,
) -> torch.Tensor:
    r"""
    Mostly copied from SentenceTransformers.

    Compute the MSE loss between the
    |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Query, Pos) - gold_sim(Query, Neg)|.
    By default, sim() is the dot-product.
    For more details, please refer to https://arxiv.org/abs/2010.02666.

    Args:
        embeddings (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, hidden_size)`):
            The three batches of embeddings (query, positives, negatives).
        labels (:obj:`torch.IntTensor` of shape :obj:`(batch_size,)`), default None:
            The labels for each triple of embeddings.
        similarity_fct (:obj:`Callable`):
            The similarity function to measure embeddings distance

    Return:
        the margin mse loss
    """

    assert len(embeddings) == 3
    assert labels is not None

    # sentence_features: query, positive passage, negative passage
    scores_pos = similarity_fct(embeddings[0], embeddings[1])
    scores_neg = similarity_fct(embeddings[0], embeddings[2])
    margin_pred = scores_pos - scores_neg

    return nn.functional.mse_loss(margin_pred, labels)


def multiple_negatives_ranking_loss(
    *embeddings: List[torch.Tensor],
    labels: torch.Tensor = None,
    scale: float = 20.0,
    similarity_fct: Callable = cos_sim,
) -> torch.Tensor:
    r"""
    Mostly copied from SentenceTransformers.

    This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
    where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.

    For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
    n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.

    This loss function works great to train embeddings for retrieval setups where you have positive pairs
    (e.g. (query, relevant_doc)) as it will sample in each batch n-1 negative docs randomly.

    The performance usually increases with increasing batch sizes.

    For more information, see: https://arxiv.org/pdf/1705.00652.pdf
    (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

    You can also provide one or multiple hard negatives per anchor-positive pair by structuring the data like this:
    (a_1, p_1, n_1), (a_2, p_2, n_2)

    Here, n_1 is a hard negative for (a_1, p_1).
    The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.

    Args:
        embeddings (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, hidden_size)`):
            The two or three batches of embeddings (query, positives) or (query, positives, negatives).
        labels (:obj:`torch.IntTensor` of shape :obj:`(batch_size,)`), default None:
            Must be None.
        scale (:obj:`float`):
            Output of similarity function is multiplied by scale value.
        similarity_fct (:obj:`Callable`):
            The similarity function to measure embeddings distance.

    Return:
        the multiple negatives ranking loss
    """

    assert len(embeddings) in (2, 3)
    assert labels is None

    embeddings_1_and_2 = torch.cat(embeddings[1:], dim=0)
    scores = similarity_fct(embeddings[0], embeddings_1_and_2) * scale

    # Example a[i] should match with b[i]
    labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
    return nn.functional.cross_entropy(scores, labels)


def multiple_negatives_symmetric_ranking_loss(
    *embeddings: List[torch.Tensor],
    labels: torch.Tensor = None,
    scale: float = 20.0,
    similarity_fct: Callable = cos_sim,
) -> torch.Tensor:
    r"""
    Mostly copied from SentenceTransformers.

    This loss is an adaptation of multiple_negatives_ranking_loss. multiple_negatives_ranking_loss computes the
    following loss: For a given anchor and a list of candidates, find the positive candidate.

    In multiple_negatives_symmetric_ranking_loss, we add another loss term: Given the positive and a list of all
    anchors, find the correct (matching) anchor.

    For the example of question-answering: You have (question, answer)-pairs. multiple_negatives_ranking_loss just
    computes the loss to find the answer for a given question. multiple_negatives_symmetric_ranking_loss additionally
    computes the loss to find the question for a given answer.

    Note: If you pass triplets, the negative entry will be ignored. A anchor is just searched for the positive.

    Args:
        embeddings (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, hidden_size)`):
            The two or three batches of embeddings (query, positives) or (query, positives, negatives).
        labels (:obj:`torch.IntTensor` of shape :obj:`(batch_size,)`), default None:
            Must be None.
        scale (:obj:`float`):
            Output of similarity function is multiplied by scale value.
        similarity_fct (:obj:`Callable`):
            The similarity function to measure embeddings distance.

    Return:
        the multiple negatives ranking loss
    """

    assert len(embeddings) in (2, 3)
    assert labels is None

    embeddings_1_and_2 = torch.cat(embeddings[1:], dim=0)
    scores = similarity_fct(embeddings[0], embeddings_1_and_2) * scale

    # Example a[i] should match with b[i]
    labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)

    anchor_positive_scores = scores[:, 0:len(embeddings[1])]
    forward_loss = nn.functional.cross_entropy(scores, labels)
    backward_loss = nn.functional.cross_entropy(anchor_positive_scores.transpose(0, 1), labels)
    return (forward_loss + backward_loss) / 2


def triplet_loss(
    *embeddings: List[torch.Tensor],
    labels: torch.Tensor = None,
    distance_metric: Callable = TripletDistanceMetric.EUCLIDEAN,
    margin: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Mostly copied from SentenceTransformers.

    This class implements triplet loss. Given a triplet of (anchor, positive, negative),
    the loss minimizes the distance between anchor and positive while it maximizes the distance
    between anchor and negative. It compute the following loss function:

    loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0).

    Margin is an important hyperparameter and needs to be tuned respectively.

    For further details, see: https://en.wikipedia.org/wiki/Triplet_loss

    Args:
        embeddings (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, hidden_size)`):
            The three batches of embeddings (query, positives, negatives).
        labels (:obj:`torch.IntTensor` of shape :obj:`(batch_size,)`), default None:
            Must be None.
        scale (:obj:`float`):
            Output of similarity function is multiplied by scale value.
        similarity_fct (:obj:`Callable`):
            The similarity function to measure embeddings distance.

    Return:
        the triplet loss
    """

    assert len(embeddings) == 3
    assert labels is None

    distance_pos = distance_metric(embeddings[0], embeddings[1])
    distance_neg = distance_metric(embeddings[0], embeddings[2])

    losses = F.relu(distance_pos - distance_neg + margin)
    return losses.mean()

import re
import string
from collections import Counter, OrderedDict, defaultdict
from typing import Dict, List, Tuple

import torch

from transformers_framework.utilities.logging import rank_zero_info


def compute_question_answering_metrics(
    all_index: List[int],
    all_start_logits: List[List[int]],
    all_end_logits: List[List[int]],
    all_tokens: List,
    all_covered_tokens: List,
    all_token_is_max_context: List,
    all_offset_mapping: List,
    all_context: List,
    all_gold_answers: List[str],
    n_best_size: int,
    max_answer_length: int,
):
    r""" Compute exact match and F1 from a series of partial (split) predictions in machine reading. """

    all_predictions = compute_question_answering_predictions(
        all_index=all_index,
        all_start_logits=all_start_logits,
        all_end_logits=all_end_logits,
        all_tokens=all_tokens,
        all_covered_tokens=all_covered_tokens,
        all_token_is_max_context=all_token_is_max_context,
        all_offset_mapping=all_offset_mapping,
        all_context=all_context,
        all_gold_answers=all_gold_answers,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
    )

    exact_raw, f1_raw = get_raw_scores(all_predictions)

    exact_match = torch.tensor(list(exact_raw.values()), dtype=torch.float).mean()
    f1 = torch.tensor(list(f1_raw.values()), dtype=torch.float).mean()

    return exact_match, f1


def compute_question_answering_predictions(
    all_index: List[int],
    all_start_logits: List[List[int]],
    all_end_logits: List[List[int]],
    all_tokens: List,
    all_covered_tokens: List,
    all_token_is_max_context: List,
    all_offset_mapping: List,
    all_context: List,
    all_gold_answers: List,
    n_best_size: int,
    max_answer_length: int,
):
    r""" Compute predictions from a series of partial (split) predictions in machine
    reading grouped by original example belonging.
    """

    # group predictions over index, which points every prediction to the same original example
    index_to_features = defaultdict(list)
    for index, *feature in zip(
        all_index,
        all_start_logits,
        all_end_logits,
        all_covered_tokens,
        all_token_is_max_context,
        all_tokens,
        all_offset_mapping,
        all_context,
        all_gold_answers,
    ):
        index_to_features[index].append(feature)

    all_predictions = OrderedDict()

    rank_zero_info("Evaluating predictions...")
    for index, features in index_to_features.items():
        preliminary_predictions = []

        for feature_index, feature in enumerate(features):
            (
                start_logit, end_logit, covered_tokens, token_is_max_context,
                tokens, offset_mapping, context, gold_answers,
            ) = feature

            start_indexes = get_best_indexes(start_logit, n_best_size=n_best_size)
            end_indexes = get_best_indexes(end_logit, n_best_size=n_best_size)

            token_is_max_context = dict(token_is_max_context)

            for start_index in start_indexes:
                if start_index >= len(tokens):
                    continue
                if start_index not in covered_tokens:
                    continue
                if not token_is_max_context.get(start_index, False):
                    continue

                for end_index in end_indexes:
                    if end_index >= len(tokens):
                        continue
                    if end_index not in covered_tokens:
                        continue
                    if end_index < start_index:
                        continue
                    if (end_index - start_index) >= max_answer_length:
                        continue

                    prelim_pred = dict(
                        feature_index=feature_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=start_logit[start_index],
                        end_logit=end_logit[end_index]
                    )
                    preliminary_predictions.append(prelim_pred)

        preliminary_predictions = sorted(
            preliminary_predictions, key=lambda x: (x['start_logit'] + x['end_logit']), reverse=True
        )

        # get exact text span of best prediction
        final_text = ""
        if preliminary_predictions:
            pred = preliminary_predictions[0]
            if pred['start_index'] > 0:
                # retrieve start and last character in original context
                start_char = offset_mapping[pred['start_index']][0]
                end_char = offset_mapping[pred['end_index']][1]
                final_text = context[start_char:end_char]
            else:
                final_text = None

        all_predictions[index] = (final_text, gold_answers)

    return all_predictions


def get_best_indexes(logits, n_best_size):
    r""" Get the n-best logits from a list. """
    index_and_scores = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    return [x[0] for x in index_and_scores[:n_best_size]]


def get_raw_scores(predictions: Dict) -> Tuple[Dict, Dict]:
    r""" Get list of EM and F1 for each question. """
    exact_scores, f1_scores = dict(), dict()

    for index, (predicted_text, gold_answers) in predictions.items():
        exact_scores[index] = metric_max_over_ground_truths(exact_match_score, predicted_text, gold_answers)
        f1_scores[index] = metric_max_over_ground_truths(f1_score, predicted_text, gold_answers)

    return exact_scores, f1_scores


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    r""" Compare prediction with every ground truth and return max pair score. """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths, default=prediction is None)


def normalize_answer(text: str):
    r""" Lower text and remove punctuation, articles and extra whitespace. """
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text.lower() if ch not in exclude)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    return text


def f1_score(prediction, ground_truth):
    r""" Compute F1 score between float predictions and labels. """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth):
    r""" Exact match requires exact prediction of text span. """
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    return int(prediction == ground_truth)

from typing import Any, List, Literal, Union

import numpy as np
import numpy.typing as npt
import torch
from lightning_fabric.utilities.distributed import _distributed_available
from numba import njit


def numpy_num_elements(array: npt.NDArray) -> int:
    r""" Return total number of elements in array. Similar to tensor.numel() in pytorch. """
    return np.prod(array.shape)


def get_rank_and_workers_dependend_seeds() -> List[int]:
    r""" Compute two seeds dependens on rank of this process and eventually also the worker id. """
    seeds = []
    if _distributed_available():
        seeds.append(torch.distributed.get_rank())
    else:
        seeds.append(0)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        seeds.append(worker_info.id)
    else:
        seeds.append(0)
    return seeds


def get_generator() -> np.random.Generator:
    # initialize generator's seed based on global_rank and worker id
    if not hasattr(get_generator, "generator"):
        setattr(get_generator, "generator", np.random.default_rng(seed=get_rank_and_workers_dependend_seeds()))
    return get_generator.generator


@njit
def unsorted_segment_sum(segment_ids: npt.NDArray[np.int64], length: int) -> npt.NDArray[np.int64]:
    r""" Equivalent of `tf.unsorted_segment_sum` with data set always to 1. """
    res = np.zeros(shape=(length,), dtype=np.int64)
    data = np.ones_like(segment_ids)

    scatter_add_1D(res, segment_ids, data)
    return res


@njit
def scatter_add_1D(
    array: npt.NDArray[np.int64],
    index: npt.NDArray[np.int64],
    data: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    r""" Scatter add for 1D dimensional arrays. """
    for i in range(index.shape[0]):
        array[index[i]] += data[i]


@njit
def correct_mask_to_cover_words(
    mask: npt.NDArray[np.bool_], word_tails: npt.NDArray[np.int64]
) -> npt.NDArray[np.bool_]:
    r""" Correct a mask such that every span starting with a tail token is shrinked/enlarged to cover only full words.

    Args:
        mask: a boolean mask over tokens
        word_tails: a boolean mask where True means tail and 0 means start of word and IGNORE_IDX for special tokens.
    """

    assert mask.shape == word_tails.shape  # nosec

    # do not work in-place
    mask = mask.copy()

    for i in range(1, len(mask)):
        if word_tails[i] == 1:  # if this is a tail token

            # if actual is masked but previous is not we are deactivating mask over tail token
            if mask[i] and not mask[i - 1]:
                mask[i] = False
            
            # if actual is not a mask but previous was, we are activating mask over tail token
            elif not mask[i] and mask[i - 1]:
                mask[i] = True

    return mask


# @njit activate when np.argpartition will be supported by numba
def adjust_sampled_array_total(array: npt.NDArray[np.int64], expected_sum: int) -> npt.NDArray[np.int64]:
    r""" Increments or decrements larger/smaller elements in array such that array.sum() == expected_sum. """
    res = array.copy()
    actual_sum = res.sum()

    while actual_sum != expected_sum:
        num_elements_to_adjust = actual_sum - expected_sum

        if num_elements_to_adjust > 0:
            # avoid out of bound index
            num_elements_to_adjust = min(num_elements_to_adjust, len(res))
            # get indexes of largest `num_elements_to_adjust` values
            indices_to_reduce = np.argpartition(res, -num_elements_to_adjust)[-num_elements_to_adjust:]
            res[indices_to_reduce] -= 1
            actual_sum -= num_elements_to_adjust

        elif num_elements_to_adjust < 0:
            # avoid out of bound index
            num_elements_to_adjust = max(num_elements_to_adjust, -len(res))
            # get indexes of smaller `num_elements_to_adjust` values
            indices_to_reduce = np.argpartition(res, -num_elements_to_adjust - 1)[:-num_elements_to_adjust]
            res[indices_to_reduce] += 1
            actual_sum -= num_elements_to_adjust

    return res


def adjust_sampled_array_max(array: npt.NDArray[np.int64], expected_max: int):
    r""" Increments or decrements larger/smaller elements in array such that array.max() <= expected_max. """
    res = array.copy()

    while res.max() > expected_max:
        elements_to_adjust_mask = (res > expected_max)
        num_elements_to_adjust = elements_to_adjust_mask.sum()

        # get indexes of smaller `num_elements_to_adjust` values
        indices_to_increase = np.argpartition(res, num_elements_to_adjust)[:num_elements_to_adjust]
        res[indices_to_increase] += 1
        res[elements_to_adjust_mask] -= 1

    return res


def random_segmentation(num_items: int, num_segments: int) -> npt.NDArray[np.int64]:
    r""" Partition a sequence of items randomly into non-empty segments.
    Follows an exponential distribution.

    Args:
        num_items: an integer scalar > 0
        num_segments: an integer scalar in [1, num_items]

    Returns:
        a Tensor with shape [num_segments] containing positive integers that add
        up to num_items
    """

    assert num_items > 0  # nosec
    assert 0 < num_segments <= num_items  # nosec

    first_in_segment = (np.arange(num_items - 1) < num_segments - 1).astype(dtype=np.int64)
    get_generator().shuffle(first_in_segment)

    # inserting a 0 at the beginning
    first_in_segment = np.insert(first_in_segment, 0, 0)

    # creates an array of the type [0, 0, 1, 1, 1, 2, 2, 3, ...]
    segment_ids = np.cumsum(first_in_segment)
    segment_length = unsorted_segment_sum(segment_ids, segment_ids.max() + 1)

    return segment_length


def poisson_segmentation(num_items: int, num_segments: int) -> npt.NDArray[np.int64]:
    r""" Partition a sequence of items randomly into non-empty segments.
    Follows a poisson distribution.

    Args:
        num_items: an integer scalar > 0
        num_segments: an integer scalar in [1, num_items]

    Returns:
        a Tensor with shape [num_segments] containing positive integers that add
        up to num_items
    """

    assert num_items > 0  # nosec
    assert 0 < num_segments <= num_items  # nosec

    mean_span_length = num_items / num_segments

    segment_length = get_generator().poisson(lam=mean_span_length, size=(num_segments, )).astype(np.int64)
    segment_length = adjust_sampled_array_total(segment_length, expected_sum=num_items)

    return segment_length


def constant_segmentation(num_items: int, num_segments: int) -> npt.NDArray[np.int64]:
    r""" Partition a sequence of items in num_segment buckets of contant size.
    
    Args:
        num_items: an integer scalar > 0
        num_segments: an integer scalar in [1, num_items]

    Returns:
        a Tensor with shape [num_segments] containing positive integers that add
        up to num_items (only if num_items was divisible by num_segments)
    """

    segment_length = int(num_items / num_segments)
    return np.full(shape=(num_segments,), fill_value=segment_length)


def random_spans_mask(
    length: int,
    probability: float = 0.15,
    mean_span_length: float = 3.0,
    max_noise_spans: int = None,
    distribution: Literal['exponential', 'poisson'] = 'exponential',
) -> npt.NDArray[np.bool_]:
    r"""
    Mostly a conversion of https://github.com/google-research/text-to-text-transfer-transformer/blob/f0cf9e8c51bd48699265763d01c2f8b24ae1098b/t5/data/preprocessors.py#L2895  # noqa
    Noise mask consisting of random spans of noise tokens.

    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:

        num_noise_tokens = round(length * probability)
        num_nonnoise_spans = num_noise_spans = round(
        num_noise_tokens / mean_span_length)

    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.

    Args:
        length: an int representing the length of the output mask
        probability: a float - approximate density of output mask
        mean_span_length: a number
        max_noise_spans: max number of spans
        distribution: whether to use exponential or poisson distribution to get spans length

    Returns:
        a list of booleans
    """

    assert distribution in ('exponential', 'poisson')  # nosec

    original_length = length
    # increase length to avoid degeneracy
    length = max(length, 2)

    num_noise_tokens = int(round(length * probability))

    # avoid degeneracy by ensuring 1 <= num_noise_tokens < length
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = int(round(num_noise_tokens / mean_span_length))
    num_noise_spans = max(num_noise_spans, 1)

    if max_noise_spans is not None:
        num_noise_spans = min(num_noise_spans, max_noise_spans)

    # create random spans for both the noise and nonnoise tokens
    if distribution == 'exponential':
        noise_span_lengths = random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = random_segmentation(num_nonnoise_tokens, num_noise_spans)
    elif distribution == 'poisson':
        noise_span_lengths = poisson_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = poisson_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )

    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = unsorted_segment_sum(span_starts, length=length)

    span_num = np.cumsum(span_start_indicator)
    mask = np.equal(span_num % 2, 1)[:original_length]

    return mask


def random_fixed_span_mask(
    length: int,
    probability: float = 0.15,
    span_length: int = 3,
    max_noise_spans: int = None,
) -> npt.NDArray[np.bool_]:
    r"""
    Similar to `random_spans_mask` but noise spans have fixed length.

    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:

        num_noise_tokens = round(length * probability)
        num_nonnoise_spans = num_noise_spans = round(
        num_noise_tokens / mean_span_length)

    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.

    Args:
        length: an int representing the length of the output mask
        probability: a float - approximate density of output mask
        span_length: a number
        max_noise_spans: max number of noise spans

    Returns:
        an array of booleans
    """

    original_length = length
    # increase length to avoid degeneracy
    length = max(length, 2)

    num_noise_tokens = int(round(length * probability))
    # avoid degeneracy by ensuring 1 <= num_noise_tokens < length
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)

    # make sure num_noise_tokens is multiple of span_length
    num_noise_tokens = span_length * int(num_noise_tokens / span_length)

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(int(round(num_noise_tokens / span_length)), 1)

    if max_noise_spans is not None:
        num_noise_spans = min(num_noise_spans, max_noise_spans)

    # create random spans for both the noise and nonnoise tokens
    noise_span_lengths = constant_segmentation(num_noise_tokens, num_noise_spans)

    # create nonnoise spans
    num_nonnoise_tokens = length - noise_span_lengths.sum()
    nonnoise_span_lengths = random_segmentation(num_nonnoise_tokens, num_noise_spans)

    # interleave noise and nonnoise spans starting with nonnoise
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )

    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = unsorted_segment_sum(span_starts, length=length)

    span_num = np.cumsum(span_start_indicator)
    mask = np.equal(span_num % 2, 1)[:original_length]

    return mask


def compress_spans_to_unique_tokens(
    tokens: npt.NDArray[np.int64], mask: npt.NDArray[np.bool_], unique_tokens: Union[int, List[int]]
) -> npt.NDArray[np.int64]:
    r""" Replace each run of consecutive noise tokens with an unique token.

    The idea here is to be able to align the dropped spans in the inputs
    with the markers in the targets.

    We want to generate training examples like
    # Original sentence: "We hold these truths to be self evident"
    # Result: "We hold X to be Y that", "X these truths Y self evident Z"

    Args:
        tokens: a 1d numpy array of token ids
        noise_mask: a boolean numpy array with the same shape as tokens
        unique_tokens: single integers or list of integers

    Returns:
        a Tensor with the same shape and dtype as tokens
    """

    prev_token_is_noise = np.roll(mask, shift=np.int64(1))
    prev_token_is_noise[0] = 0

    first_noise_tokens = np.logical_and(mask, np.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = np.logical_and(mask, prev_token_is_noise)

    sentinels = np.zeros_like(first_noise_tokens, dtype=np.int64)
    num_noise_tokens = first_noise_tokens.sum()

    if isinstance(unique_tokens, int):
        unique_tokens = np.ones(shape=(num_noise_tokens,), dtype=np.int64) * unique_tokens
    else:
        unique_tokens = np.array(unique_tokens)[:num_noise_tokens]

    sentinels[first_noise_tokens] = unique_tokens

    tokens = np.where(first_noise_tokens, sentinels, tokens)
    tokens = tokens[~subsequent_noise_tokens]
    return tokens


def numpy_multinomial(
    probabilities: npt.NDArray[np.float32],
    generator: np.random.Generator = None,
) -> npt.NDArray[np.int64]:
    r""" Sample `num_samples` elements from a multinomial distribution. """
    if generator is None:
        generator = np.random.default_rng()

    positions = generator.multinomial(1, pvals=probabilities)
    return positions.nonzero()[1]


@njit
def numpy_softmax(z, axis: int = -1):
    num = np.exp(z)
    s = num / np.sum(np.exp(z), axis)
    return s


@njit
def numpy_min_max_softmax_normalization(array: npt.NDArray[np.float32], beta: float = 2.0):
    r"""Apply min-max norm followed by a softmax. """

    # target_clusters_counts has shape (number_of_substituted_tokens, n_clusters)
    minimum = np.array([np.min(a) for a in array])
    maximum = np.array([np.max(a) for a in array])

    # normalize distribution over target clusters
    denominator = (maximum - minimum)
    denominator[denominator == 0] = 1

    min_max_norm_array = (array - minimum) / denominator
    return numpy_softmax(min_max_norm_array * beta, axis=-1)


def pad_numpy_sequence(
    array: npt.NDArray,
    padding_value: Any,
    length: int,
    truncate: bool = True,
    padding_side: Literal['right', 'left'] = 'right',
    dtype: npt.DTypeLike = None,
) -> List:
    r""" Pad an array with values up to length either on right or left. """
    assert array.ndim == 1

    if len(array) > length:
        return array[:length] if truncate else array
    else:
        padding_size = length - len(array)
        if padding_side == 'right':
            array = np.concatenate([array, [padding_value] * padding_size], axis=0)
        else:
            np.concatenate([[padding_value] * padding_size, array], axis=0)

    if dtype is not None:
        array = array.astype(dtype)

    return array

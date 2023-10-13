from typing import Any, Dict, Generator, Iterable, List, Literal, Mapping, Tuple, Union

import numpy as np
import numpy.typing as npt


def split(_list: Iterable, part_length: int = None, num_parts: int = None) -> Iterable:
    r""" Split an Iterable `_list` in parts of length `part_length`.
    Eventually drop last piece if it would have been shorter. """

    if (part_length is None) == (num_parts is None):
        raise ValueError("You must define either `part_length` or `num_parts` when calling `split`")

    if not isinstance(_list, Iterable):
        raise ValueError("`_list` must be an iterable")

    # split given the length of each part
    if part_length is not None:
        if not isinstance(part_length, int) or not part_length > 0:
            raise ValueError("`part_length` must be a positive integer")

        for i in range(0, len(_list), part_length):
            yield _list[i:i + part_length]

    # split given the total number of parts
    else:
        if not isinstance(num_parts, int) or not num_parts > 0:
            raise ValueError("`num_parts` must be a positive integer")

        _list = list(_list)
        k, m = divmod(len(_list), num_parts)

        yield from (_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_parts))


def shrink_batch(
    batch: Dict[str, Union[Any, npt.NDArray]],
    keys: List[str],
    pad_token_id: int,
    shrink_to_multiples_of: int = None,
):
    r""" Remove data on the sequence length dimension in the positions where every example is padded. """
    if not keys:
        return

    assert all(k not in batch or (isinstance(batch[k], np.ndarray) and batch[k].ndim == 2) for k in keys)  # nosec

    # compute positions to remove along sequence length (dim 1)
    indexes = (batch[keys[0]] != pad_token_id).any(axis=0)
    if shrink_to_multiples_of is not None:
        original_indexes_shape = indexes.shape
        indexes = indexes.reshape(-1, shrink_to_multiples_of).any(axis=-1, keepdims=True)
        indexes = np.tile(indexes, (1, shrink_to_multiples_of)).reshape(original_indexes_shape)

    # do shrinking
    for key in keys:
        batch[key] = batch[key][:, indexes]


def pad_array(
    array: npt.NDArray,
    padding_value: int,
    length: int,
    truncate: bool = True,
    padding_side: Literal['right', 'left'] = 'right'
) -> npt.NDArray:
    r""" Pad an array with values up to length either on right or left. """

    assert array.ndim == 1, "pad_array works with one dimensional arrays"  # nosec

    if len(array) > length:
        return array[:length] if truncate else array
    else:
        padding_size = length - len(array)
        pad_width = [(0, padding_size)] if padding_side == 'right' else [(padding_size, 0)]
        array = np.pad(array, pad_width=pad_width, mode='constant', constant_values=padding_value)

    return array


def pad_sequence(
    array: Iterable,
    padding_value: Any,
    length: int,
    truncate: bool = True,
    padding_side: Literal['right', 'left'] = 'right'
) -> List:
    r""" Pad an array with values up to length either on right or left. """

    if len(array) > length:
        return array[:length] if truncate else array
    else:
        padding_size = length - len(array)
        if padding_side == 'right':
            array = array + [padding_value] * padding_size
        else:
            array = [padding_value] * padding_size + array

    return array


def shuffle_lists(*lists: Tuple[List], generator: np.random.Generator) -> Tuple[List]:
    r""" Shuffle many lists with the same indexes. """
    if not lists:
        return tuple()
    assert all(len(_l) == len(lists[0]) for _l in lists)

    tmp = list(zip(*lists))
    generator.shuffle(tmp)
    return tuple(list(t) for t in zip(*tmp))


def sort_lists(*lists: Tuple[List], indexes: List[int] = None) -> Tuple[List]:
    r""" Sort many lists with the same indexes. """
    if not lists:
        return tuple()
    assert all(len(_l) == len(lists[0]) for _l in lists)
    assert sorted(indexes) == list(range(len(lists[0]))), (
        f"Indexes does not match lists to be sorted: len indexes: {len(indexes)}, "
        f"len: lists: {len(lists[0])}. elements in indexes: {indexes}"
    )

    res = tuple([_l[i] for i in indexes] for _l in lists)
    return res


def argsort_list(_list: List, descending: bool = False) -> List[int]:
    r""" Get sort indexes for list. """
    return [i[0] for i in sorted(enumerate(_list), key=lambda x:x[1], reverse=descending)]


def shift_tokens_right(
    input_ids: npt.NDArray[np.int64], pad_token_id: int, decoder_start_token_id: int
) -> npt.NDArray[np.int64]:
    r""" Shift input ids one token to the right. Copied from BART and converted to numpy. """

    assert pad_token_id is not None, "pad token id must be not None"  # nosec

    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[1:] = input_ids[:-1].copy()
    shifted_input_ids[0] = decoder_start_token_id

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids[shifted_input_ids == -100] = pad_token_id

    return shifted_input_ids


def special_zip(*iterators, stop: str = 'shortest') -> Iterable:
    r""" Zip allowing None iterators (which will be threated as infinite None generators).
    If argument `stop` is `longest`, exhausted generators will yield None until the longest is finished.
    """
    assert stop in ('shortest', 'longest'), f"`stop` argument can only be `shortest` or `longest`, got {stop}"  # nosec

    def inf_gen():
        while True:
            yield None

    iterators = [iter(iterator) if iterator is not None else inf_gen() for iterator in iterators]

    if stop == 'shortest':
        yield from zip(*iterators)
    else:
        while True:
            data = [next(iterator, None) for iterator in iterators]
            if all(x is None for x in data):
                break
            yield data


def add_dict_to_attributes(obj: object, dictionary: Dict):
    for k, v in dictionary.items():
        setattr(obj, k, v)


def multi_get_from_dict(dictionary, *keys, default: Any = None):
    r""" Get many keys from dictionary without having to index multiple times.
    Returns `default` is key is not found.
    """
    return (dictionary.get(k, default) for k in keys)


def dict2list(data: Dict[Any, Iterable]) -> Iterable[Dict]:
    r""" Convert a dict of lists to a list of dicts. """

    # get all the data and assert each value is list
    values = list(data.values())
    assert all(isinstance(v, Iterable) for v in values)

    # assert each value has same length to be able to create list of small dicts
    assert all(len(v) == len(values[0]) for v in values)

    if not data or any(len(v) == 0 for v in values):
        return []

    # create output dictionary using the same keys for all entries
    keys = data.keys()
    res = [dict(zip(keys, values)) for values in zip(*[data[key] for key in keys])]
    return res


def list2dict(data: Iterable[Dict]) -> Dict[Any, Iterable]:
    r""" Convert a list of dicts to a dict of lists. """

    data = list(data)

    if not data:
        return {}

    # check all instances in the input list are dicts
    assert all(isinstance(d, Mapping) for d in data)

    # check all input dicts have the same keys
    keys = data[0].keys()
    assert all(d.keys() == keys for d in data)

    # merge data
    res = {k: [d[k] for d in data] for k in keys}
    return res


def get_position_score(doc_span_start, doc_span_length, position):
    r""" Get score of a token in a span. The score represents the level of context available for that token. """
    return min(
        position - doc_span_start,  # left context
        (doc_span_start + doc_span_length - 1) - position  # right context
    ) + 0.001 * doc_span_length


def check_is_max_context(all_sequence_ids, doc_stride: int):
    r""" Check if this is the 'max context' doc span for the token. """

    # extract spans sequence
    doc_token_spans = []
    last_span_length = doc_stride  # fake previous span just to avoid if-else below
    relative_start_position = 0

    for sequence_ids in all_sequence_ids:
        segment_length = sum(int(s == 1) for s in sequence_ids)

        # update start of relative (from start of second span) and absolute (from beginning of input ids) position
        relative_start_position += (last_span_length - doc_stride)
        absolute_start_position = sequence_ids.index(1) + relative_start_position

        last_span_length = segment_length
        doc_token_spans.append((relative_start_position, absolute_start_position, segment_length))

    # get largest interval to create numpy array
    start_position = min(doc_span_start for doc_span_start, _, _ in doc_token_spans)
    end_position = max(doc_span_start + doc_span_length for doc_span_start, _, doc_span_length in doc_token_spans)

    # each element should be the max_context vector of each span
    res = np.zeros(shape=(len(doc_token_spans), end_position - start_position))

    # for every position in every span compute the context score
    for span_index, (doc_span_start, _, doc_span_length) in enumerate(doc_token_spans):
        for position in range(doc_span_start, doc_span_start + doc_span_length):
            res[span_index, position - start_position] = get_position_score(doc_span_start, doc_span_length, position)

    # find best position along each span
    output = np.equal(res, res.max(axis=0, keepdims=True)).astype(int).tolist()

    # return
    for o, (doc_span_start, abs_start_position, doc_span_length) in zip(output, doc_token_spans):
        interval = o[doc_span_start - start_position:doc_span_start + doc_span_length - start_position]
        yield {i + abs_start_position: value for i, value in enumerate(interval)}


def filter_generator(generator_in: Generator, step: int = 1, offset: int = 0) -> Generator:
    r"""
    Return elements from a generator. First `offset` elements are discarded
    Then, return an element after every every `step` extracted
    """

    assert step is not None and step >= 0, f"step must be non-negative, found {step}"  # nosec
    assert offset is not None and offset >= 0, f"offset must be non-negative, found {offset}"  # nosec

    # advance to the target offset and return first element
    for _ in range(offset):
        try:
            next(generator_in)
        except StopIteration:
            return
    try:
        yield next(generator_in)
    except StopIteration:
        return

    while True:
        # consume world_size - 1 inputs
        for _ in range(step - 1):
            try:
                next(generator_in)
            except StopIteration:
                return
        try:
            yield next(generator_in)
        except StopIteration:
            return


def batch_filter(generator_in: Generator, size: int = 1) -> Generator:
    r"""
    By reading `size` elements at a time, we assure that no last iteration will have
    a different batch size across nodes, that would cause a fail.
    """
    assert size >= 0, f"Cannot read {size} elements at a time. size must be >= 0"
    while True:
        res = []
        for i in range(size):
            try:
                res.append(next(generator_in))
            except StopIteration:
                return
        for i in res:
            yield i


def split_dict_on_prefixes(dictionary: Dict[str, Any], prefixes: List[str], remove_prefix: bool = True):
    r""" Split the keys and values in multiple dictionaries based on key prefixes.
    The number of returned dictionaries if equal to len(prefixes) + 1, with the latter dict containing
    everything that didn't match at least a prefix. Keys are compared to prefixes in the given order,
    with a first-match policy. """

    res = dict(**dictionary)
    others = []
    for prefix in prefixes:
        tmp = dict()
        for k in list(res.keys()):
            if k.startswith(prefix):
                new_k = k[len(prefix):] if remove_prefix else k
                tmp[new_k] = res.pop(k)
        others.append(tmp)

    return others + [res]

from multiprocessing import cpu_count
from typing import Any, Dict, List, Literal

from datasets import Dataset, Sequence
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers_framework.utilities import IGNORE_IDX
from transformers_framework.utilities.datasets import datasets_disable_caching
from transformers_framework.utilities.functional import (
    argsort_list,
    dict2list,
    list2dict,
    pad_sequence,
    shuffle_lists,
    sort_lists,
    split,
)
from transformers_framework.utilities.numpy import get_generator
from transformers_framework.utilities.processors import convert_examples_to_features, process_entry_question_answering


def prepare_dataset_question_answering(
    dataset: Dataset,
    query_column: str,
    context_column: str,
    answers_column: str,
    label_column: str,
    max_sequence_length: int,
    doc_stride: int,
    max_query_length: int,
    tokenizer: PreTrainedTokenizerBase,
    load_from_cache_file: bool = True,
    num_workers: int = cpu_count(),
    batch_size: int = 1000,
) -> Dataset:
    r""" Process dataset, possibly taking advantage of multiprocessing and caching. """
    dataset = dataset.map(
        process_entry_question_answering,
        num_proc=num_workers,
        remove_columns=dataset.column_names,
        load_from_cache_file=load_from_cache_file,
        with_indices=True,
        desc="Preprocessing examples",
        fn_kwargs=dict(
            query_column=query_column,
            context_column=context_column,
            answers_column=answers_column,
            label_column=label_column,
            tokenizer=tokenizer,
        ),
    )

    dataset = dataset.map(
        convert_examples_to_features,
        num_proc=num_workers,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
        desc="Tokenizing examples",
        load_from_cache_file=load_from_cache_file,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
        )
    )

    return dataset


generator = get_generator()


def _group_fn(
    examples: Dict[str, List[Any]],
    question_column: str = None,
    answer_column: str = None,
    index_column: str = None,
    label_column: str = None,
    k: int = None,
    pad: bool = True,
    selection: Literal['best', 'worst'] = None,
    scores_column: str = None,
    grouping: str = Literal['random', 'fixed'],
):
    # tranform from dict of lists to list of dicts
    examples = dict2list(examples)

    res = []
    for example in examples:

        if selection is not None:
            indexes = argsort_list(example[scores_column], descending=True)

            example[index_column], example[answer_column], example[label_column] = sort_lists(
                example[index_column], example[answer_column], example[label_column], indexes=indexes,
            )

            if selection == 'best':
                example[index_column], example[answer_column], example[label_column] = (
                    example[index_column][:k], example[answer_column][:k], example[label_column][:k]
                )
            else:
                example[index_column], example[answer_column], example[label_column] = (
                    example[index_column][-k:], example[answer_column][-k:], example[label_column][-k:]
                )

        if grouping == 'random':
            example[index_column], example[answer_column], example[label_column] = shuffle_lists(
                example[index_column], example[answer_column], example[label_column], generator=generator,
            )

        for idx, answers, labels in zip(
            split(example[index_column], part_length=k),
            split(example[answer_column], part_length=k),
            split(example[label_column], part_length=k),
        ):
            if pad:
                idx = pad_sequence(idx, -1, k, truncate=True, padding_side='right')
                # pad_sequence(answers, "", k, truncate=True, padding_side='right')
                labels = pad_sequence(labels, IGNORE_IDX, k, truncate=True, padding_side='right')

            res.append({
                question_column: example[question_column],
                index_column: idx,
                answer_column: answers,
                label_column: labels,
            })

    res = list2dict(res)
    return res


def answer_selection_grouping(
    dataset: Dataset,
    input_columns: str = None,
    index_column: str = None,
    label_column: str = None,
    k: int = None,
    num_workers: int = cpu_count(),
    batch_size: int = None,
    group: bool = None,
    pad: bool = True,
    grouping: str = Literal['random', 'fixed'],
    selection: Literal['best', 'worst'] = None,
    scores_column: str = None,
    load_from_cache_file: bool = True,
):
    assert k is not None, f"k should be not None in `answer_selection_grouping`, got {k}"
    assert len(input_columns) == 2, f"`answer_selection_grouping` works only with 2 input columns, got {input_columns}"
    assert load_from_cache_file is False, (
        "`answer_selection_grouping` needs `load_from_cache_file=False` to create random groupings"
    )
    assert (selection is None) == (scores_column is None), (
        "`answer_selection_grouping` need either both `selection` and `scores_column` or none of the two"
    )
    if scores_column is not None or selection is not None:
        assert scores_column in dataset.column_names
        assert selection in ('best', 'worst')

    question_column, answer_column = input_columns

    # check whether grouping is needed
    if group is None:
        group = not isinstance(dataset.features[answer_column], Sequence)

    # group dataset on question column
    if group:
        dataset = group_dataset_over_column(dataset, pivot_column=question_column)

    fn_kwargs = dict(
        question_column=question_column,
        answer_column=answer_column,
        index_column=index_column,
        label_column=label_column,
        k=k,
        pad=pad,
        grouping=grouping,
        selection=selection,
        scores_column=scores_column,
    )

    # avoid caching on this specific map because it should output
    # every time a different sequence of answers for each question
    with datasets_disable_caching():
        dataset = dataset.map(
            _group_fn,
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=load_from_cache_file,
            fn_kwargs=fn_kwargs,
            num_proc=num_workers,
            remove_columns=dataset.column_names,
        )

    return dataset


def group_dataset_over_column(
    dataset: Dataset,
    pivot_column: str = None,
):
    """ Group dataset on pivot column. """
    index = dict()
    other_columns = dataset.column_names
    other_columns.remove(pivot_column)

    for example in tqdm(dataset, desc="Grouping dataset..."):
        if example[pivot_column] not in index:
            index[example[pivot_column]] = dict()
        for other_column in other_columns:
            if other_column not in index[example[pivot_column]]:
                index[example[pivot_column]][other_column] = []
            index[example[pivot_column]][other_column].append(example[other_column])

    index = [{pivot_column: q, **v} for q, v in index.items()]
    return Dataset.from_list(index)

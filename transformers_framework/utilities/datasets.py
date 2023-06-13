import os
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset, disable_caching, enable_caching, load_dataset
from datasets.builder import DatasetGenerationError
from datasets.fingerprint import logger

from transformers_framework.utilities.format import partial_format
from transformers_framework.utilities.logging import rank_zero_error, rank_zero_info, rank_zero_warn


def load_dataset_from_huggingface(path: str, config: str = None, keep_in_memory: bool = False) -> Dataset:
    r""" Load dataset from HF repository.

    Raises:
        `FileNotFoundError` if the remote repository does not exist.
    """

    try:
        # `load_dataset` may raise FileNotFoundError if HF repo does not exist or `DatasetGenerationError` if
        # a path with the same name is present on disk
        path, split = os.path.dirname(path), os.path.basename(path)
        dataset = load_dataset(path, name=config, split=split, keep_in_memory=keep_in_memory)
    except DatasetGenerationError:
        raise FileNotFoundError

    return dataset


def load_dataset_from_local(path: str, keep_in_memory: bool = False) -> Dataset:
    r""" Load dataset from local repository. The dataset must have been saved with `Dataset.save_to_disk(...)`.
    
    Raises:
        `FileNotFoundError` if the remote repository does not exist.
    """

    dataset = Dataset.load_from_disk(path, keep_in_memory=keep_in_memory)
    return dataset


def load_dataset_from_s3(path: str, keep_in_memory: bool = False) -> Dataset:
    r""" Load dataset from S3 repository. The dataset must have been saved with `Dataset.save_to_disk(...)`.

    Raises:
        `FileNotFoundError` if the remote repository does not exist.
    """

    dataset = Dataset.load_from_disk(path, keep_in_memory=keep_in_memory)
    return dataset


def load_dataset_from_anywhere(
    path: str, config: str = None, shard: int = None, keep_in_memory: bool = False
) -> Tuple[Dataset, str]:
    r"""
    Loads a dataset from:
        - HF repository
        - disk dump saved with `Dataset.save_to_disk(...)`
        - remote S3 dump saved with `Dataset.save_to_disk(..., storage_options="s3")`
    """

    if path.startswith("s3://"):
        try:
            dataset = load_dataset_from_s3(path=path, keep_in_memory=keep_in_memory)
            location = "s3"
        except FileNotFoundError:
            rank_zero_error(f"'{path}' is not a valid S3 path")
            exit(1)
    else:
        try:
            dataset = load_dataset_from_local(path=path, keep_in_memory=keep_in_memory)
            location = "disk"
        except FileNotFoundError:
            try:
                dataset = load_dataset_from_huggingface(path=path, config=config, keep_in_memory=keep_in_memory)
                location = "huggingface"
            except FileNotFoundError:
                rank_zero_error(f"'{path}' is not a valid dataset path on disk or HF")
                exit(1)

    if config is not None and location != "huggingface":
        rank_zero_warn("You loaded a dataset from S3 or from disk and used config parameters, which is not required.")

    if shard is not None and shard > 1:
        dataset = dataset.shard(shard, 0, writer_batch_size=100000, contiguous=False)
    
    return dataset, location


def reduce(
    dataset,
    column: str,
    batch_size: Optional[int] = 1000,
    remove_columns: Optional[Union[str, List[str]]] = None,
    keep_in_memory: bool = False,
    load_from_cache_file: bool = None,
    cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
    disable_nullable: bool = False,
    num_proc: Optional[int] = None,
    suffix_template: str = "_{rank:05d}_of_{num_proc:05d}_iter_{iter:05d}",
    desc: Optional[str] = None
) -> Dataset:
    r"""
    Reduce all the examples in the table (individually or in batches) given a pivot column and update the table.
    If your function returns a column that already exists, then it overwrites it.

    You can specify whether the function should be batched or not with the `batched` parameter:

    Args:
        column (`str`):
            Columns on which the dataset will be reduced. Examples with the same value on `column` will
            see their other columns concatenated in a list.
        batch_size (`int`, *optional*, defaults to `2`):
            Process batch of examples instead of just a single pair. This is transparent to the user but may
            increase speed and memory usage.
        remove_columns (`Optional[Union[str, List[str]]]`, defaults to `None`):
            Remove a selection of columns while doing the mapping.
            Columns will be removed before updating the examples with the output of `function`, i.e.
            if `function` is adding columns with names in `remove_columns`, these columns will be kept.
        keep_in_memory (`bool`, defaults to `False`):
            Keep the dataset in memory instead of writing it to a cache file.
        load_from_cache_file (`bool`, defaults to `True` if caching is enabled):
            If a cache file storing the current computation from `function`
            can be identified, use it instead of recomputing.
        cache_file_name (`str`, *optional*, defaults to `None`):
            Provide the name of a path for the cache file. It is used to store the
            results of the computation instead of the automatically generated cache file name.
        writer_batch_size (`int`, defaults to `1000`):
            Number of rows per write operation for the cache file writer.
            This value is a good trade-off between memory usage during the processing, and processing speed.
            Higher value makes the processing do fewer lookups, lower value consume less temporary memory
            while running `map`.
        disable_nullable (`bool`, defaults to `False`):
            Disallow null values in the table.
        num_proc (`int`, *optional*, defaults to `None`):
            Max number of processes when generating cache. Already cached shards are loaded sequentially.
        suffix_template (`str`):
            If `cache_file_name` is specified, then this suffix will be added at the end of the base name of each.
            Defaults to `"_{rank:05d}_of_{num_proc:05d}_iter_{iter:05d}"`.
            For example, if `cache_file_name` is "processed.arrow", then for `rank=1` and `num_proc=4`, the resulting
            file would be `"processed_00001_of_00004.arrow"` for the default suffix.
        desc (`str`, *optional*, defaults to `None`):
            Meaningful description to be displayed alongside with the progress bar while mapping examples.

    Example:

    ```python
    >>> ds = CustomDataset.load_dataset("lucadiliello/wikiqa")
    >>> def merge_on_question(example_1, example_2):
    ...     res = {k: [example_1[k], example_2[k]] for k in example_1.keys()}
    ...     return res
    >>> ds = ds.reduce(merge_on_question)
    >>> ds[0]
    ```
    """

    if batch_size < 2:
        raise ValueError("required `batch_size` greater or equal then 2")

    if keep_in_memory and cache_file_name is not None:
        raise ValueError("Please use either `keep_in_memory` or `cache_file_name` but not both.")

    if num_proc is not None and num_proc <= 0:
        raise ValueError("num_proc must be an integer > 0.")

    # If the array is empty we do nothing
    # (but we make sure to handle an empty indicesmapping and remove the requested columns anyway)
    if len(dataset) == 0:
        if dataset._indices is not None:  # empty indices mapping
            dataset = Dataset(
                dataset.data.slice(0, 0),
                info=dataset.info.copy(),
                split=dataset.split,
            )
        if remove_columns:
            return dataset.remove_columns(remove_columns)
        else:
            return dataset

    def prepare_function(examples: Dict[str, Any]) -> Dict[str, List[Any]]:
        return {key: [[v] for v in values] for key, values in examples.items()}

    def reduce_function(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # group over columns other than the main `column`

        other_column_names = [other_column for other_column in examples.keys() if other_column != column]
        res = {k: [] for k in examples.keys()}

        last_main_value = None
        for main_column_data, *other_column_data in zip(*[examples[k] for k in [column] + other_column_names]):
            if main_column_data != last_main_value:
                res[column].append(main_column_data)
                last_main_value = main_column_data
                for key in other_column_names:
                    res[key].append([])
            for column_name, data in zip(other_column_names, other_column_data):
                res[column_name][-1] += data

        return res

    if isinstance(remove_columns, str):
        remove_columns = [remove_columns]

    if column is None or column not in dataset._data.column_names:
        raise ValueError(
            f"Column {column} not in the dataset. Current columns in the dataset: {dataset._data.column_names}"
        )

    if remove_columns is not None and column in remove_columns:
        raise ValueError("Main reduce column {column} must not be in `remove_columns`")

    if remove_columns is not None and any(col not in dataset._data.column_names for col in remove_columns):
        raise ValueError(
            f"Column to remove {list(filter(lambda col: col not in dataset._data.column_names, remove_columns))} "
            f"not in the dataset. Current columns in the dataset: {dataset._data.column_names}"
        )

    if num_proc is not None:
        num_proc = min(num_proc, len(dataset))

    if remove_columns:
        initial_dataset = dataset.remove_columns(remove_columns)
    else:
        initial_dataset = dataset

    logger.info("Computing length of final dataset")
    target_dataset_length = len(dataset.unique(column=column))

    # prepare columns different from `column` for merging
    prepared_dataset = initial_dataset.map(
        prepare_function,
        with_indices=False,
        with_rank=False,
        batch_size=batch_size,
        batched=True,
        drop_last_batch=False,
        remove_columns=None,
        keep_in_memory=keep_in_memory,
        load_from_cache_file=load_from_cache_file,
        cache_file_name=cache_file_name,
        writer_batch_size=writer_batch_size,
        disable_nullable=disable_nullable,
        num_proc=num_proc,
        suffix_template=partial_format(suffix_template, iter=0),
        desc=desc,
    )

    # sort dataset on `column`
    sorted_dataset = prepared_dataset.sort(
        column=column, load_from_cache_file=load_from_cache_file, writer_batch_size=writer_batch_size
    )

    index = 1
    while len(sorted_dataset) > target_dataset_length:
        sorted_dataset = sorted_dataset.map(
            reduce_function,
            with_indices=False,
            with_rank=False,
            batched=True,
            batch_size=batch_size,
            drop_last_batch=False,
            remove_columns=None,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            disable_nullable=disable_nullable,
            num_proc=num_proc,
            suffix_template=partial_format(suffix_template, iter=index),
            desc=desc,
        )
        batch_size = batch_size ** 2  # need to increase bs to avoid getting stuck
        index += 1

        if num_proc is not None:
            num_proc = min(num_proc, len(dataset))

    return sorted_dataset


class datasets_disable_caching(object):

    def __enter__(self):
        rank_zero_info("Temporarily disabling caching for datasets")
        disable_caching()

    def __exit__(self, exc_type, exc_val, exc_tb):
        rank_zero_info("Re-enabling caching for datasets")
        enable_caching()

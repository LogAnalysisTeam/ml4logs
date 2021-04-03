from datasets import load_from_disk, Dataset, concatenate_datasets
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from transformers import AutoTokenizer
import logging
from time import time
import numpy as np

from dataset_utils import tokenize_no_special_tokens, prepare_target_and_context_from_chunk_indices

log = logging.getLogger(__name__)


def chunkify_list_in_memory(data: List, chunk_size: int, drop_last_incomplete_chunk:bool = True):
    chunked = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    if drop_last_incomplete_chunk and len(chunked[-1]) != chunk_size:
        del chunked[-1]
    return chunked


def tokenize_dataset(ds: Dataset,
                     output_ds_basedir: Path,
                     tokenizer_bert_model: str, 
                     save_dataset: bool=True) -> Dataset:
    output_path = output_ds_basedir / f'purely_tokenized_{tokenizer_bert_model}'
    log.info(f'Tokenizing dataset to {output_path}')
    start = time()
    try:
        output_ds = load_from_disk(output_path)
    except FileNotFoundError as e:
        log.info(f"No tokenized cache found, computing")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_bert_model)
        output_ds = ds.map(tokenize_no_special_tokens, 
                           fn_kwargs={'tokenizer': tokenizer}, 
                           batched=True, 
                           batch_size=100000, 
                           writer_batch_size=10000)
        if save_dataset:
            log.info(f"Saving tokenized, time taken so far: {time() - start}s, saving to {output_path}")
            output_ds.save_to_disk(output_path)
    log.info(f'Tokenization done, time taken: {time() - start}s')
    return output_ds


def chunkify_dataset_in_memory_return_dict(ds: Dataset,
                                           chunk_size: int,
                                           drop_last_incomplete_chunk:bool = True) -> Dict[str, List[List[Any]]]:
    log.info(f"Creating contexts of size {chunk_size} in memory")
    start = time()
    tokens_chunks = chunkify_list_in_memory(ds['tokens'], chunk_size, drop_last_incomplete_chunk=drop_last_incomplete_chunk)
    log.info(f'Tokens chunked, time taken so far: {time() - start}s')
    text_chunks = chunkify_list_in_memory(ds['text'], chunk_size, drop_last_incomplete_chunk=drop_last_incomplete_chunk)
    log.info(f'Text chunked, time taken: {time() - start}s')
    return {'chunk_text': text_chunks,
            'chunk_tokens': tokens_chunks}


def get_train_val_sizes(total_size, val_ratio):
    val_size = int(np.ceil(total_size*val_ratio))
    train_size = total_size - val_size
    assert train_size + val_size == total_size
    return train_size, val_size


def get_train_val_indices(length: int,
                          desired_total_taken: int,
                          val_ratio: float,
                          rnd: np.random.Generator,
                          sorted_indices: bool=False):
    total_to_take = min(length, desired_total_taken)
    train_size, val_size = get_train_val_sizes(total_to_take, val_ratio)
    indices = rnd.choice(length, size=total_to_take, replace=False)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    if sorted_indices:
        train_indices = np.sort(train_indices)
        val_indices = np.sort(val_indices)
    return train_indices, val_indices


def train_val_split_in_memory_dict(in_memory_data: Dict[str, List[Any]],
                                   desired_total_taken:int,
                                   val_ratio: float,
                                   rnd: np.random.Generator) -> Dict[str, Dict[str, List[Any]]]:
    log.info(f"Creating train val split with val_ratio: {val_ratio}")
    length = len(next(iter(in_memory_data.values())))
    train_indices, val_indices = get_train_val_indices(length, desired_total_taken, val_ratio, rnd)
    log.info(f'Train size: {len(train_indices)}, validation size: {len(val_indices)}, total size: {len(train_indices) + len(val_indices)} from desired count {desired_total_taken}')
    train_dict = {key: [values[idx] for idx in train_indices] for key, values in in_memory_data.items()}
    val_dict = {key: [values[idx] for idx in val_indices] for key, values in in_memory_data.items()}
    return {'train': train_dict,
            'val': val_dict}


def create_dataset_from_in_memory_dict(in_memory_data: Dict[str, List[Any]], 
                                       output_path: Path,
                                       save_dataset: bool=True) -> Dataset:
    log.info(f'Creating dataset from in memory dict to {output_path}')
    start = time()
    in_memory_ds = Dataset.from_dict(in_memory_data)
    on_disk_ds = in_memory_ds.map(function=None, writer_batch_size=50000, batched=True, batch_size=5000)
    log.info(f'In memory dict mapped to on disk time taken so far: {time() - start}s{f", saving to {output_path}" if save_dataset else ""}')
    if save_dataset:
        on_disk_ds.save_to_disk(output_path)
    log.info(f'{f"Saved, t" if save_dataset else "T"}otal time taken: {time() - start}s, saved to {output_path}')
    return on_disk_ds


def chunkify_train_val_split(ds: Dataset,
                             output_ds_basedir: Path,
                             chunk_size: int,
                             desired_total_chunks_taken:int,
                             val_ratio: float,
                             seed: int,
                             drop_last_incomplete_chunk:bool = True,
                             save_dataset: bool=True) -> Dict[str, Dataset]:
    output_basepath = output_ds_basedir / f'chunked-size-{chunk_size}_split-total-size-{desired_total_chunks_taken}_seed-{seed}'
    log.info(f'Chunking and splitting to {output_basepath}')
    start = time()
    train_path = output_basepath / 'train'
    val_path = output_basepath / 'val'
    try:
        train_ds = load_from_disk(train_path)
        val_ds = load_from_disk(val_path)
    except FileNotFoundError as e:
        chunked_dict = chunkify_dataset_in_memory_return_dict(ds, chunk_size)
        train_val_split_dict = train_val_split_in_memory_dict(chunked_dict, desired_total_chunks_taken, val_ratio,
                                                              rnd=np.random.default_rng(seed=seed))
        del chunked_dict
        train_ds = create_dataset_from_in_memory_dict(train_val_split_dict['train'], train_path, save_dataset=save_dataset)
        val_ds = create_dataset_from_in_memory_dict(train_val_split_dict['val'], val_path, save_dataset=save_dataset)
        del train_val_split_dict
    log.info(f"Chunking and splitting done, time taken: {time() - start}s")
    return {'train': train_ds,
            'val': val_ds}


def tokenize_chunkify_split_pipeline_single_dataset(ds: Dataset,
                                                    output_ds_basedir: Path,
                                                    tokenizer_bert_model: str,
                                                    chunk_size: int,
                                                    desired_total_chunks_taken:int,
                                                    val_ratio: float,
                                                    seed: int,
                                                    drop_last_incomplete_chunk:bool = True,
                                                    save_dataset: bool=True,
                                                    ds_name_for_logging: str = None
                                                    ) -> Dict[str, Dataset]:
    cur_name = output_ds_basedir.stem if ds_name_for_logging is None else ds_name_for_logging
    log.info(f"Tokenization chunkification and split pipeline for {cur_name}")
    start = time()
    tokenized_ds = tokenize_dataset(ds, output_ds_basedir, tokenizer_bert_model, save_dataset=save_dataset)
    train_val_split = chunkify_train_val_split(tokenized_ds, output_ds_basedir, chunk_size, desired_total_chunks_taken, val_ratio, seed,
                                               drop_last_incomplete_chunk=drop_last_incomplete_chunk, save_dataset=save_dataset)
    log.info(f'Done tok-chunk-split pipeline for {cur_name}, time taken: {time()- start}s')
    return train_val_split


def combine_datasets(datasets: Dict[str, Dataset],
                     output_path: Path,
                     seed: int,
                     save_dataset: bool=True):
    try:
        combined_ds = load_from_disk(output_path)
    except FileNotFoundError as e:
        log.info(f"Concatenating datasets")
        start = time()
        unshuffled_combined_ds = concatenate_datasets(list(datasets.values()))
        log.info(f"Shuffling datasets, time taken so far: {time() - start}")
        combined_ds = unshuffled_combined_ds.shuffle(seed=seed).flatten_indices(writer_batch_size=20000)
        log.info(f'Shuffled dataset, time taken so far: {time() - start}')
        if save_dataset:
            log.info(f'Saving to {output_path}')
            combined_ds.save_to_disk(output_path)
        log.info(f"Combining time taken: {time() - start}")
    return combined_ds


def prepare_targets_contexts_dataset(ds: Dataset,
                                     output_ds_basedir: Path,
                                     epochs: int,
                                     seed: int,
                                     remove_target_prob: float,
                                     chunk_target_context_columns: List[Tuple[str, str, str]] = None,
                                     num_proc: Optional[int] = None,
                                     save_dataset: bool=True) -> Dataset:
    """
    Assumes epochs is smaller than the chunk size and that all chunks are the same size
    """
    if chunk_target_context_columns is None:
        chunk_target_context_columns = [('chunk', 'target', 'context')]
    output_path = output_ds_basedir / f'targets_contexts_non_flat_epochs-{epochs}_seed-{seed}'
    log.info(f'Creating targets and contexts to {output_path}')
    start = time()
    try:
        output_ds = load_from_disk(output_path)
        log.info(f'Cached targets contexts found')
    except FileNotFoundError as e:
        chunk_size = len(ds[0][chunk_target_context_columns[0][0]])  # assumes all chunks same size
        chunk_count = len(ds)
        rnd = np.random.default_rng(seed=seed)
        target_indices_per_chunk = [rnd.choice(chunk_size, replace=False, size=epochs, shuffle=False) for _ in range(chunk_count)]
        remove_target_per_chunk = rnd.random(size=(chunk_count, epochs)) < remove_target_prob
        output_ds = ds.map(function=prepare_target_and_context_from_chunk_indices,
                           with_indices=True,
                           fn_kwargs={
                               'target_indices_per_chunk': target_indices_per_chunk,
                               'remove_target_per_chunk' : remove_target_per_chunk,
                               'chunk_target_context_columns': chunk_target_context_columns
                           },
                           batched=True,
                           batch_size=1000,
                           writer_batch_size=10000,
                           remove_columns=ds.column_names,
                           num_proc=num_proc)
        if save_dataset:
            log.info(f"Time taken so far: {time() - start}s, Saving to {output_path}")
            output_ds.save_to_disk(output_path)
    log.info(f'Creating targets contexts time taken: {time() - start}s')
    return output_ds


def flatten_contexts_in_dataset(ds: Dataset,
                                output_ds_basedir: Path,
                                flatten_truncate_batch_map_function,
                                truncation_type: str,
                                num_proc: Optional[int] = None,
                                keep_columns: Optional[List[str]] = None,
                                save_dataset: bool=True) -> Dataset:
    output_path = output_ds_basedir / f'flattened_contexts_truncation-{truncation_type.replace(" ", "_")}'
    log.info(f'Flattening with {truncation_type} truncation to {output_path}')
    start = time()
    try:
        output_ds = load_from_disk(output_path)
        log.info(f'Cached flattened truncated found')
    except FileNotFoundError as e:
        if keep_columns is not None:
            keep_columns = set(keep_columns)
        remove_columns = None if keep_columns is None else [column for column in ds.column_names if column not in keep_columns]
        output_ds = ds.map(function=flatten_truncate_batch_map_function,
                           batched=True,
                           writer_batch_size=10000,
                           remove_columns=remove_columns,
                           num_proc=num_proc)
        if save_dataset:
            log.info(f'Time taken so far: {time() - start}s, saving to {output_path}')
            output_ds.save_to_disk(output_path)

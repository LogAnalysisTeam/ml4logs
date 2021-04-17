import numpy as np
from typing import List, Tuple, Any, Dict
import re
from pathlib import Path
from collections import defaultdict
import shutil

from datasets import config as datasets_config
from datasets import load_from_disk, fingerprint


HDFS1_TIMESTAMP_PATTERN = re.compile(r'^(\d+) (\d+) (\d+) ')
HDFS1_BLK_ID_PATTERN = re.compile(r'(blk_-?\d+)')


def my_caching_load_from_disk(path):
    ds_info_path = path / "dataset_info.json"
    state_path = path / "state.json"
    arrow_files = list(path.glob("*.arrow"))
    if ds_info_path.exists() and state_path.exists() and arrow_files:
        temp_dir_path = datasets_config.HF_DATASETS_CACHE / "tmp_datasets" / f'{path.stem}_{fingerprint.generate_random_fingerprint()}'
        temp_dir_path.mkdir(parents=True, exist_ok=False)
        files_to_copy = [ds_info_path, state_path] + arrow_files
        for file_path in files_to_copy:
            shutil.copy2(file_path, temp_dir_path)
        return load_from_disk(temp_dir_path)
    else:
        # Will throw error, but use huggingface load to throw good FileNotFoundError
        return load_from_disk(path)


def load_hdfs1_log_file_grouped(log_path: Path) -> Dict:
    blocks = defaultdict(list)
    with log_path.open(mode='r') as f:
        for line in f:
            block_id = HDFS1_BLK_ID_PATTERN.search(line).group()
            blocks[block_id].append(line)
    return blocks


def create_target_and_processed_context_deterministic(context: List[Any], target_idx: int, remove_target: bool):
    target = context[target_idx]
    processed_context = context[:target_idx] + context[target_idx + remove_target:]
    return target, processed_context


def create_target_and_processed_context(context: List[List[int]], rnd: np.random.Generator, remove_target_prob:float):
    target_idx = rnd.integers(low=0, high=len(context))
    remove_target = rnd.random() < remove_target_prob
    return create_target_and_processed_context_deterministic(context, target_idx, remove_target)


def flatten_context_simple(context: List[List[int]]) -> List[int]:
    return [token for sentence in context for token in sentence]


def flatten_and_truncate_helper(context: List[List[int]], truncate_lengths: List[int]) -> List[int]:
    return [token for (sentence, truncate_length) in zip(context, truncate_lengths) for token in sentence[:truncate_length]]


def create_target_and_flat_context(context: List[List[int]], rnd: np.random.Generator, remove_target_prob:float):
    target_sentence, processed_context = create_target_and_processed_context(context, rnd, remove_target_prob)
    flattened_context = flatten_context_simple(processed_context)
    return target_sentence, flattened_context


def prepare_target_and_context(examples, epochs, rnd: np.random.Generator, remove_target_prob:float):
    targets = []
    processed_contexts = []
    for context in examples['chunk']:
        for _ in range(epochs):
            t, c = create_target_and_processed_context(context, rnd, remove_target_prob)
            targets.append(t)
            processed_contexts.append(c)
    return {'target': targets,
            'context': processed_contexts}


def prepare_target_and_context_from_chunk_indices(examples,
                                                  indices: List[int],
                                                  target_indices_per_chunk: List[List[int]], 
                                                  remove_target_per_chunk: List[List[bool]], 
                                                  chunk_target_context_columns: List[Tuple[str, str, str]] = None):
    if chunk_target_context_columns is None:
        chunk_target_context_columns = [('chunk', 'target', 'context')]
    output_dict = {}
    for chunk_col, target_col, context_col in chunk_target_context_columns:
        targets = []
        processed_contexts = []
        for chunk, chunk_idx in zip(examples[chunk_col], indices):
            for target_idx, remove_target in zip(target_indices_per_chunk[chunk_idx], remove_target_per_chunk[chunk_idx]):
                target, context = create_target_and_processed_context_deterministic(chunk, target_idx, remove_target)
                targets.append(target)
                processed_contexts.append(context)
        output_dict[target_col] = targets
        output_dict[context_col] = processed_contexts
    return output_dict


def compute_mean_truncate_lengths(lenghts: List[int], max_length:int) -> List[int]:
    if sum(lenghts) <= max_length:
        return lenghts
    else:
        avg_len = max_length//len(lenghts)
        return [min(length, avg_len) for length in lenghts]


def compute_smart_mean_truncate_lengths(lengths: List[int], max_length: int) -> List[int]:
    if sum(lengths) <= max_length:
        return lengths
    else:
        np_lengths = np.array(lengths)
        prev_thresh = -1
        cur_thresh = max_length//len(lengths)
        while cur_thresh != prev_thresh:
            #  find the highest threshold such that the sets of list lengths smaller than it and bigger than it don't change
            prev_thresh = cur_thresh
            mask_where_slack = (np_lengths < cur_thresh) # find positions where length smaller than threshold
            non_slack_truncate = (max_length-np_lengths[mask_where_slack].sum())//(~mask_where_slack).sum() # compute truncation lengths only for positions greater than current threshold, distributing them to fit (max_length - sum(length of lists shorter than current threshold))
            cur_thresh = non_slack_truncate
        intermediate_lengths = np_lengths.copy()
        intermediate_lengths[~mask_where_slack] = np.minimum(np_lengths[~mask_where_slack], non_slack_truncate)
        safe_to_add = (max_length-np_lengths[mask_where_slack].sum())%(~mask_where_slack).sum()  # find how much is guaranteed to have been unused
        indices = np.nonzero(~mask_where_slack)[0]  # find indices where the lists are longer or same than threshold
        indices_where_can_add = indices[intermediate_lengths[indices] < np_lengths[indices]]  # find indices which are still not at full capacity
        final_indices = indices_where_can_add[:safe_to_add]  # try to take safe_to_add as many of those indices to increase their selected lengths by one
        final_lengths = intermediate_lengths.copy()
        final_lengths[final_indices] = np.minimum(intermediate_lengths[final_indices]+1, np_lengths[final_indices])
        return final_lengths.tolist()


def compute_concat_to_max_len_truncate_lengths(lenghts: List[int], max_length: int) -> List[int]:
    if sum(lenghts) <= max_length:
        return lenghts
    else:
        np_lengths = np.array(lenghts)
        need_to_remove = np.maximum(np.cumsum(np_lengths) - max_length, 0)
        new_lengths = np.maximum(np_lengths - need_to_remove, 0)
        return new_lengths.tolist()


def flatten_truncate_function_creator(truncate_lengths_function):
    def flatten_truncate(context: List[List[int]], max_length: int) -> List[int]:
        sentence_lenghts = [len(sentence) for sentence in context]
        truncate_lengths = truncate_lengths_function(sentence_lenghts, max_length)
        return flatten_and_truncate_helper(context, truncate_lengths)
    return flatten_truncate


def flatten_truncate_map_wrapper(flatten_truncate_function, max_length:int, input_column:str, output_column: str):
    def flatten_wrapper(example):
        return {output_column: flatten_truncate_function(example[input_column], max_length)}
    return flatten_wrapper


def flatten_truncate_batch_map_wrapper(flatten_truncate_function, max_length:int, input_column:str, output_column: str):
    def flatten_wrapper(batch):
        return {output_column: [flatten_truncate_function(example, max_length) for example in batch[input_column]]}
    return flatten_wrapper


def flatten_contexts_wrapper(example):
    return {'flat_context': flatten_context_simple(example['context'])}


def flatten_contexts_batch(examples):
    return {'flat_context': [flatten_context_simple(context) for context in examples['context']]}


def prepare_ict(examples, epochs, rnd: np.random.Generator, remove_target_prob:float):
    targets = []
    flat_contexts = []
    for context in examples['chunk']:
        for _ in range(epochs):
            t, f = create_target_and_flat_context(context, rnd, remove_target_prob)
            targets.append(t)
            flat_contexts.append(f)
    return {'target': targets,
            'flat_context': flat_contexts}


def remove_timestamp_old(example):
    # need to find third occurence of a space and slice the string after it
    # using a very non robust silly solution
    s = example['text']
    example['text'] = s[s.find(' ', s.find(' ', s.find(' ')+1)+1)+1:]
    return example


def remove_timestamp(example):
    example['text'] = HDFS1_TIMESTAMP_PATTERN.sub('', example['text'])
    return example


def chunkify(examples):
    return {"chunk": [examples['tokens']]}


def tokenize_no_special_tokens(examples, tokenizer):
    return {'tokens': tokenizer(examples['text'], add_special_tokens=False, truncation=True, return_attention_mask=False)['input_ids']}
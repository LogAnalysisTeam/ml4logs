import numpy as np
from typing import List, Tuple, Any
import re

HDFS1_TIMESTAMP_PATTERN = re.compile(r'^(\d+) (\d+) (\d+) ')


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
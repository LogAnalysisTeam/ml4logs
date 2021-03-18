import numpy as np
from typing import List
import re

HDFS1_TIMESTAMP_PATTERN = re.compile(r'^(\d+) (\d+) (\d+) ')


def create_target_and_processed_context(context: List[List[int]], rnd: np.random.Generator, remove_target_prob:float):
    target_idx = rnd.integers(low=0, high=len(context))
    remove_target = rnd.random() < remove_target_prob
    target_sentence = context[target_idx]
    processed_context = context[:target_idx] + context[target_idx + remove_target:]
    return target_sentence, processed_context


def flatten_context_simple(context: List[List[int]]) -> List[int]:
    return [token for sentence in context for token in sentence]


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
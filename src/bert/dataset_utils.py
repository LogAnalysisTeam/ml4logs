import numpy as np
from typing import List


def create_target_and_flat_context(context: List[List[int]], rnd: np.random.Generator, remove_target_prob:float):
    target_idx = rnd.integers(low=0, high=len(context))
    remove_target = rnd.random() < remove_target_prob
    target_sentence = context[target_idx]
    processed_context = context[:target_idx] + context[target_idx + remove_target:]
    flattened_context = [token for sentence in context for token in sentence]
    return target_sentence, flattened_context


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


def remove_timestamp(example):
    # need to find third occurence of a space and slice the string after it
    # using a very non robust silly solution
    s = example['text']
    example['text'] = s[s.find(' ', s.find(' ', s.find(' ')+1)+1)+1:]
    return example


def chunkify(examples):
    return {"chunk": [examples['tokens']]}


def tokenize_no_special_tokens(examples, tokenizer):
    return {'tokens': tokenizer(examples['text'], add_special_tokens=False, truncation=True, return_attention_mask=False)['input_ids']}
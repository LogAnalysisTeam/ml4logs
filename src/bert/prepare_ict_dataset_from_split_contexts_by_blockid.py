from datasets import load_dataset, load_from_disk, Dataset
from pathlib import Path
from typing import List, DefaultDict
import numpy as np
import time
from transformers import AutoTokenizer
from collections import defaultdict
import re

from dataset_utils import prepare_target_and_context, remove_timestamp, flatten_contexts_wrapper


def group_by_block_ids(lines: List[str]) -> DefaultDict:
    traces = defaultdict(list)
    regex = re.compile(r'(blk_-?\d+)')  # pattern eg. blk_-1608999687919862906
    for line in lines:
        block_id = regex.search(line).group()
        traces[block_id].append(line)
    return traces


def create_contexts_from_blocks(block_dict, min_context_size:int, max_context_size:int):
    contexts = []
    for block_id, lines in block_dict.items():
        cur_context = []
        for line in lines:
            cur_context.append(line)
            if len(cur_context) >= max_context_size:
                contexts.append(cur_context)
                cur_context = []
        if min_context_size <= len(cur_context) < max_context_size:
            contexts.append(cur_context)
    return contexts


def prepare_dataset_from_log_file_contexts_by_block_id(config):
    assert config.dataset_log_file is not None, "There must be a log file"
    assert config.threads >= 1, "Threads must be atleast 1"
    raw_dataset_log_path = Path(config.dataset_log_file)
    data_basedir_path = Path(config.data_basedir)

    log_filename_stem = raw_dataset_log_path.stem
    INFO = f'Epochs-{config.epochs}_Seed-{config.seed}_ByBlockId_MinContext-{config.min_context_size}_MaxContext-{config.max_context_size}'
    NAME = f'{log_filename_stem}_{INFO}'
    print(NAME)
    print(config)

    interim_path = data_basedir_path / 'interim' / log_filename_stem

    processed_path = data_basedir_path / 'processed' / NAME

    if config.tokenized_chunked_dataset_checkpoint is None:
        print(f"Loading raw dataset from {raw_dataset_log_path}")
        start = time.time()
        raw_dataset = load_dataset('text', data_files=str(raw_dataset_log_path), split='train')
        print("Raw loaded")
        print(f'Time taken: {time.time() - start}s')

        print(f'Removing timestamps')
        start = time.time()
        cleaned_dataset = raw_dataset.map(remove_timestamp, num_proc=config.threads)
        cleaned_path = interim_path / 'removed_timestamps'
        print(f'Removed, saving to {cleaned_path}')
        cleaned_dataset.save_to_disk(cleaned_path)
        print(f'Time taken: {time.time() - start}s')


        print("Grouping lines by block ids")
        start = time.time()
        blocks = group_by_block_ids(cleaned_dataset['text'])
        print("Grouped, in memory")
        print(f'Time taken: {time.time() - start}s')

        print(f"Creating chunks from blocks with min size {config.min_context_size}, max {config.max_context_size}")
        start = time.time()
        chunked_untokenized_dataset = Dataset.from_dict({'chunk_strings': create_contexts_from_blocks(blocks, config.min_context_size, config.max_context_size)})
        chunked_untokenized_path = interim_path / f'untokenized_chunked_by_block_ids_min{config.min_context_size}_max{config.max_context_size}'
        print(f"Chunked by block ids, saving to {chunked_untokenized_path}")
        chunked_untokenized_dataset.save_to_disk(chunked_untokenized_path)
        print(f'Time taken: {time.time() - start}s')


        print(f"Tokenizing dataset (per chunk)")
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
        chunked_tokenized_dataset = chunked_untokenized_dataset.map(lambda example: {'chunk': tokenizer(example['chunk_strings'], add_special_tokens=False, truncation=True, return_attention_mask=False)['input_ids'] },
                                                                    remove_columns=['chunk_strings'])
        del tokenizer
        
        chunked_tokenized_path = interim_path / f'tokenized_chunked_by_block_ids_min{config.min_context_size}_max{config.max_context_size}'
        print(f'Tokenized, saving to {chunked_tokenized_path}')
        chunked_tokenized_dataset.save_to_disk(chunked_tokenized_path)
        print(f'Time taken: {time.time() - start}s')

        del blocks
        del chunked_untokenized_dataset
    else:
        print(f"Loading chunked tokenized checkpoint from {config.tokenized_chunked_dataset_checkpoint}")
        chunked_tokenized_dataset = load_from_disk(config.tokenized_chunked_dataset_checkpoint)

    print(f"Creating targets and non-flat contexts for {config.epochs} epochs")
    start = time.time()
    rnd = np.random.default_rng(config.seed)
    targets_contexts_dataset = chunked_tokenized_dataset.map(prepare_target_and_context,
                                                             fn_kwargs={'epochs': config.epochs,
                                                                        'rnd': rnd, 
                                                                        'remove_target_prob': config.remove_target_percentage},
                                                             batched=True,
                                                             batch_size=500,
                                                             remove_columns=chunked_tokenized_dataset.column_names)
    targets_contexts_path = interim_path / f'targets_contexts_{INFO}'
    print(f"Targets and contexts done, saving to {targets_contexts_path}")
    targets_contexts_dataset.save_to_disk(targets_contexts_path)
    print(f'Time taken: {time.time() - start}s')

    print("Flattening contexts")
    start = time.time()
    finalized_dataset = targets_contexts_dataset.map(flatten_contexts_wrapper, 
                                                     remove_columns=['context'],
                                                     num_proc=config.threads).shuffle(seed=config.seed)
    print(f"Done, saving to {processed_path}")
    finalized_dataset.save_to_disk(processed_path)
    print(f'Time taken: {time.time() - start}s')
            


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dataset preparation")
    parser.add_argument("--remove-target-percentage", default=0.9, type=float)
    parser.add_argument('--bert-model', default="distilbert-base-cased", type=str, help="Pretrained Transformer for the encoder towers.")
    parser.add_argument("--min-context-size", default=3, type=int)
    parser.add_argument("--max-context-size", default=10, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--seed", default=43, type=int)
    parser.add_argument("--data-basedir", default="/home/cernypro/dev/source/ml4logs/data", type=str)
    parser.add_argument("--dataset-log-file", default=None, type=str)
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--tokenized-chunked-dataset-checkpoint", default=None, type=str)

    config = parser.parse_args()
    prepare_dataset_from_log_file_contexts_by_block_id(config)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time taken: {end - start}s')
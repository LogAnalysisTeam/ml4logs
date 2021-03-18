from datasets import load_dataset, load_from_disk
from pathlib import Path
from typing import List
import numpy as np
import time
from transformers import AutoTokenizer

from dataset_utils import prepare_target_and_context, remove_timestamp, chunkify, tokenize_no_special_tokens, flatten_contexts_wrapper

def prepare_dataset_from_log_file(config):
    assert config.dataset_log_file is not None, "There must be a log file"
    assert config.threads >= 1, "Threads must be atleast 1"
    raw_dataset_log_path = Path(config.dataset_log_file)
    data_basedir_path = Path(config.data_basedir)

    log_filename_stem = raw_dataset_log_path.stem
    INFO = f'Epochs-{config.epochs}_Seed-{config.seed}'
    NAME = f'{log_filename_stem}_{INFO}'
    print(NAME)
    print(config)

    interim_path = data_basedir_path / 'interim' / log_filename_stem

    processed_path = data_basedir_path / 'processed' / NAME


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

    print(f"Tokenizing dataset")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
    tokenized_dataset = cleaned_dataset.map(tokenize_no_special_tokens, fn_kwargs={'tokenizer': tokenizer}, batched=True, batch_size=100000)
    del tokenizer
    
    tokenized_path = interim_path / 'purely_tokenized'
    print(f'Tokenized, saving to {tokenized_path}')
    tokenized_dataset.save_to_disk(tokenized_path)
    print(f'Time taken: {time.time() - start}s')

    print(f"Creating contexts of size {config.context_sentence_count}")
    start = time.time()
    contexts_dataset = tokenized_dataset.map(chunkify,
                                             batched=True,
                                             batch_size=config.context_sentence_count,
                                             drop_last_batch=True,
                                             remove_columns=tokenized_dataset.column_names,
                                             num_proc=config.threads)
    chunked_path = interim_path / f'chunked_size_{config.context_sentence_count}'
    print(f'Chunked, saving to {chunked_path}')
    contexts_dataset.save_to_disk(chunked_path)
    print(f'Time taken: {time.time() - start}s')


    print(f"Creating targets and non-flat contexts for {config.epochs} epochs")
    start = time.time()
    rnd = np.random.default_rng(config.seed)
    targets_contexts_dataset = contexts_dataset.map(prepare_target_and_context,
                                             fn_kwargs={'epochs': config.epochs,
                                                        'rnd': rnd, 
                                                        'remove_target_prob': config.remove_target_percentage},
                                             batched=True,
                                             batch_size=500,
                                             remove_columns=contexts_dataset.column_names)
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
    parser.add_argument("--context-sentence-count", default=10, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--seed", default=43, type=int)
    parser.add_argument("--data-basedir", default="/home/cernypro/dev/source/ml4logs/data", type=str)
    parser.add_argument("--dataset-log-file", default=None, type=str)
    parser.add_argument("--threads", default=1, type=int)

    config = parser.parse_args()
    prepare_dataset_from_log_file(config)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time taken: {end - start}s')
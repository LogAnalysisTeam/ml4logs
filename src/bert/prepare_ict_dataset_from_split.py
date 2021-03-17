from datasets import load_dataset, load_from_disk
from pathlib import Path
from typing import List
import numpy as np
import time
from transformers import AutoTokenizer

from dataset_utils import prepare_ict, remove_timestamp, chunkify, tokenize_no_special_tokens

def prepare_dataset_from_log_file(config):
    assert config.dataset_log_file is not None, "There must be a log file"
    raw_dataset_log_path = Path(config.dataset_log_file)
    data_basedir_path = Path(config.data_basedir)

    log_filename_stem = raw_dataset_log_path.stem
    NAME = f'{log_filename_stem}_Epochs-{config.epochs}_Seed-{config.seed}'
    print(NAME)
    print(config)

    interim_path = data_basedir_path / 'interim' / log_filename_stem

    processed_path = data_basedir_path / 'processed' / NAME


    print(f"Loading raw dataset from {raw_dataset_log_path}")
    raw_dataset = load_dataset('text', data_files=str(raw_dataset_log_path), split='train')
    print("Raw loaded")

    print(f'Removing timestamps')
    cleaned_dataset = raw_dataset.map(remove_timestamp, num_proc=4)
    cleaned_path = interim_path / 'removed_timestamps'
    print(f'Removed, saving to {cleaned_path}')
    cleaned_dataset.save_to_disk(cleaned_path)

    print(f"Tokenizing dataset")
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
    tokenized_dataset = cleaned_dataset.map(tokenize_no_special_tokens, fn_kwargs={'tokenizer': tokenizer}, batched=True, batch_size=100000)
    del tokenizer
    
    tokenized_path = interim_path / 'purely_tokenized'
    print(f'Tokenized, saving to {tokenized_path}')
    tokenized_dataset.save_to_disk(tokenized_path)

    print(f"Creating contexts of size {config.context_sentence_count}")
    contexts_dataset = tokenized_dataset.map(chunkify,
                                             batched=True,
                                             batch_size=config.context_sentence_count,
                                             drop_last_batch=True,
                                             remove_columns=tokenized_dataset.column_names,
                                             num_proc=4)
    chunked_path = interim_path / f'chunked_size_{config.context_sentence_count}'
    print(f'Chunked, saving to {chunked_path}')
    contexts_dataset.save_to_disk(chunked_path)

    rnd = np.random.default_rng(config.seed)

    print(f"Final ICT preparation, creating targets and contexts for {config.epochs} epochs")
    finalized_dataset = contexts_dataset.map(prepare_ict,
                                             fn_kwargs={'epochs': config.epochs,
                                                        'rnd': rnd, 
                                                        'remove_target_prob': config.remove_target_percentage},
                                             batched=True,
                                             batch_size=500,
                                             remove_columns=contexts_dataset.column_names).shuffle(seed=config.seed)
    print(f"Done, saving to {processed_path}")
    finalized_dataset.save_to_disk(processed_path)
            


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

    config = parser.parse_args()
    prepare_dataset_from_log_file(config)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Seconds taken: {end - start}')
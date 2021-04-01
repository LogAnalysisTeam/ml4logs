from datasets import load_dataset, load_from_disk
from pathlib import Path
from typing import List
import time
from transformers import AutoTokenizer

from dataset_utils import prepare_target_and_context, tokenize_no_special_tokens


def chunkify_tokens_and_text(examples):
    return {'chunk_text' : [examples['text']], 'chunk_tokens': [examples['tokens']]}


def tokenize_and_chunkify(config):
    assert config.dataset_log_file is not None, "There must be a log file"
    assert config.threads >= 1, "Threads must be atleast 1"
    raw_dataset_log_path = Path(config.dataset_log_file)
    output_basedir_path = Path(config.output_basedir)

    log_filename_stem = raw_dataset_log_path.stem
    INFO = f'Context-Size-{config.context_sentence_count}'
    NAME = f'{log_filename_stem}_{INFO}'
    print(NAME)
    print(config)

    output_dir_path = output_basedir_path / log_filename_stem

    print(f"Loading raw dataset from {raw_dataset_log_path}")
    start = time.time()
    raw_dataset = load_dataset('text', data_files=str(raw_dataset_log_path), split='train', keep_in_memory=True)
    print("Raw loaded")
    print(f'Time taken: {time.time() - start}s')

    print(f"Tokenizing dataset")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
    tokenized_dataset = raw_dataset.map(tokenize_no_special_tokens, fn_kwargs={'tokenizer': tokenizer}, batched=True, batch_size=100000, keep_in_memory=True)
    del tokenizer
    del raw_dataset
    
    tokenized_path = output_dir_path / 'purely_tokenized'
    print(f'Tokenized, saving to {tokenized_path}')
    tokenized_dataset.save_to_disk(tokenized_path)
    print(f'Time taken: {time.time() - start}s')

    print(f"Creating contexts of size {config.context_sentence_count}")
    start = time.time()
    contexts_dataset = tokenized_dataset.map(chunkify_tokens_and_text,
                                             batched=True,
                                             batch_size=config.context_sentence_count,
                                             drop_last_batch=True,
                                             remove_columns=tokenized_dataset.column_names,
                                             num_proc=config.threads,
                                             keep_in_memory=True)
    chunked_path = output_dir_path / f'chunked_size_{config.context_sentence_count}_tokens_text'
    del tokenized_dataset
    print(f'Chunked, saving to {chunked_path}')
    contexts_dataset.save_to_disk(chunked_path)
    print(f'Time taken: {time.time() - start}s')
            


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tokenization and chunkification")
    parser.add_argument('--bert-model', default="distilbert-base-cased", type=str, help="Pretrained Transformer for tokenizer.")
    parser.add_argument("--context-sentence-count", default=10, type=int)
    parser.add_argument("--output-basedir", default="/home/cernypro/dev/source/ml4logs/data", type=str)
    parser.add_argument("--dataset-log-file", default=None, type=str)
    parser.add_argument("--threads", default=1, type=int)

    config = parser.parse_args()
    tokenize_and_chunkify(config)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time taken: {end - start}s')
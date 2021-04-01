from datasets import load_dataset, load_from_disk, Dataset
from pathlib import Path
from typing import List
import time
from transformers import AutoTokenizer
import numpy as np

from dataset_utils import tokenize_no_special_tokens

def chunkify_in_memory(data: List, chunk_size: int, drop_last_incomplete_chunk:bool = True):
    chunked = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    if drop_last_incomplete_chunk and len(chunked[-1]) != chunk_size:
        del chunked[-1]
    return chunked


def tokenize_and_chunkify(config):
    assert config.dataset_log_file is not None, "There must be a log file"
    raw_dataset_log_path = Path(config.dataset_log_file)
    output_basedir_path = Path(config.output_basedir)

    log_filename_stem = raw_dataset_log_path.stem
    INFO = f'Context-Size-{config.context_sentence_count}'
    NAME = f'{log_filename_stem}_{INFO}'
    print(NAME)
    print(config)

    output_dir_path = output_basedir_path / log_filename_stem
    if config.tokenized_checkpoint is None:
        print(f"Loading raw dataset from {raw_dataset_log_path}")
        start = time.time()
        raw_dataset = load_dataset('text', data_files=str(raw_dataset_log_path), split='train')
        print("Raw loaded")
        print(f'Time taken: {time.time() - start}s')

        print(f"Tokenizing dataset")
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
        tokenized_dataset = raw_dataset.map(tokenize_no_special_tokens, fn_kwargs={'tokenizer': tokenizer}, batched=True, batch_size=100000, writer_batch_size=50000)
        del tokenizer
        
        tokenized_path = output_dir_path / f'purely_tokenized_{config.bert_model}'
        print(f'Tokenized, saving to {tokenized_path}')
        tokenized_dataset.save_to_disk(tokenized_path)
        print(f'Time taken: {time.time() - start}s')
    else:
        print(f"Loading tokenized checkpoint from{config.tokenized_checkpoint}")
        start = time.time()
        tokenized_dataset = load_from_disk(config.tokenized_checkpoint)
        print(f'Time taken: {time.time() - start}s')

    print(f"Creating contexts of size {config.context_sentence_count}")
    start = time.time()
    text_list = tokenized_dataset['text']
    tokens_list = tokenized_dataset['tokens']
    assert len(text_list) == len(tokens_list)
    text_chunks = chunkify_in_memory(text_list, config.context_sentence_count, drop_last_incomplete_chunk=not config.keep_incomplete_chunk)
    tokens_chunks = chunkify_in_memory(tokens_list, config.context_sentence_count, drop_last_incomplete_chunk=not config.keep_incomplete_chunk)
    assert len(text_chunks) == len(tokens_chunks)

    del tokenized_dataset

    contexts_dataset = Dataset.from_dict({'chunk_text': text_chunks, 'chunk_tokens': tokens_chunks})
    chunked_path = output_dir_path / f'chunked_size_{config.context_sentence_count}_tokens_text'

    rnd = np.random.default_rng()
    rand_num = rnd.integers(low=0, high=np.iinfo(np.int64).max, endpoint=True)
    temp_arrow_file_path = Path('/data/temporary/temp_tokenizing_chunking_in_memory') / f'{log_filename_stem}_{rand_num}.arrow'
    temp_arrow_file_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Chunked, None mapping to cache file {temp_arrow_file_path}')
    print(f'Time so far {time.time() - start}s')

    

    contexts_dataset.map(function=None, cache_file_name=str(temp_arrow_file_path), writer_batch_size=50000, batched=True, batch_size=5000)
    print(f"Mapped, time so far {time.time() - start}s")
    del contexts_dataset
    contexts_dataset = Dataset.from_file(str(temp_arrow_file_path))
    print(f"From file, time so far {time.time() - start}s, saving now")
    contexts_dataset.save_to_disk(chunked_path)
    print(f'Time taken: {time.time() - start}s')
            


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tokenization and chunkification in memory")
    parser.add_argument('--bert-model', default="distilbert-base-cased", type=str, help="Pretrained Transformer for tokenizer.")
    parser.add_argument("--context-sentence-count", default=10, type=int)
    parser.add_argument("--output-basedir", default="/home/cernypro/dev/source/ml4logs/data", type=str)
    parser.add_argument("--dataset-log-file", default=None, type=str)
    parser.add_argument('--keep-incomplete-chunk', default=False, action='store_true', help="When creating context, keep even chunk of different size than the rest")
    parser.add_argument('--tokenized-checkpoint', default=None, type=str)
    parser.add_argument("--threads", default=1, type=int)

    config = parser.parse_args()
    tokenize_and_chunkify(config)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time taken: {end - start}s')
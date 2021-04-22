from datasets import load_dataset, load_from_disk
from pathlib import Path
from typing import List
import time
from transformers import AutoTokenizer
import logging
import os

from dataset_pipeline import tokenize_chunkify_whole_single_dataset

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

log = logging.getLogger(__name__)

def tokenize_and_chunkify(config):
    assert config.dataset_log_file is not None, "There must be a log file"
    assert config.threads >= 1, "Threads must be atleast 1"
    raw_dataset_log_path = Path(config.dataset_log_file)

    log_filename_stem = raw_dataset_log_path.stem
    INFO = f'Context-Size-{config.context_sentence_count}'
    NAME = f'{raw_dataset_log_path.stem}_{INFO}'
    log.info(NAME)
    log.info(config)

    output_dir_path = raw_dataset_log_path.parent / raw_dataset_log_path.stem

    log.info(f"Loading raw dataset from {raw_dataset_log_path}")
    start = time.time()
    raw_dataset = load_dataset('text', data_files=str(raw_dataset_log_path), split='train')
    log.info("Raw loaded")
    log.info(f'Time taken: {time.time() - start}s')

    chunked_ds = tokenize_chunkify_whole_single_dataset(raw_dataset, output_dir_path,
                                                        config.bert_model, config.context_sentence_count, 
                                                        drop_last_incomplete_chunk=True, save_dataset=True)
            


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tokenization and chunkification")
    parser.add_argument('--bert-model', default="distilbert-base-cased", type=str, help="Pretrained Transformer for tokenizer.")
    parser.add_argument("--context-sentence-count", default=10, type=int)
    parser.add_argument("--dataset-log-file", default=None, type=str)
    parser.add_argument("--threads", default=1, type=int)

    config = parser.parse_args()
    tokenize_and_chunkify(config)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    log.info(f'Total time taken: {end - start}s')
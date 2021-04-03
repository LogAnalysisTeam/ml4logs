from datasets import load_dataset, load_from_disk
from pathlib import Path
from typing import List
import numpy as np
import time
from transformers import AutoTokenizer

import logging
import os

from dataset_pipeline import prepare_targets_contexts_dataset, flatten_contexts_in_dataset
from dataset_utils import flatten_truncate_batch_map_wrapper, flatten_truncate_function_creator, compute_mean_truncate_lengths
from milp_dataset_utils import compute_uniform_truncate_lenghts

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

log = logging.getLogger(__name__)

def prepare_dataset_from_chunks(config):
    assert config.dataset is not None, "There must be a log file"
    assert config.output_basedir is not None
    assert config.threads >= 1, "Threads must be atleast 1"
    chunked_dataset_path = Path(config.dataset)
    output_basedir_path = Path(config.output_basedir)
    NAME = f'Pipeline - creating ICT and flatten from {chunked_dataset_path} to basedir {output_basedir_path}'
    log.info(NAME)
    log.info(config)

    chunks_ds = load_from_disk(chunked_dataset_path)

    chunk_target_context_columns = [
        ('chunk_text', "target_text", "context_text"),
        ('chunk_tokens', 'target', 'context')
    ]

    targets_contexts_ds = prepare_targets_contexts_dataset(chunks_ds,
                                                           output_basedir_path,
                                                           epochs=config.epochs,
                                                           seed=config.seed,
                                                           remove_target_prob=config.remove_target_percentage,
                                                           chunk_target_context_columns=chunk_target_context_columns,
                                                           num_proc=config.threads,
                                                           save_dataset=True)

    flatten_truncate_function = flatten_truncate_function_creator(compute_uniform_truncate_lenghts if config.uniform_truncation else compute_mean_truncate_lengths)
    flatten_truncate_batch_map_func = flatten_truncate_batch_map_wrapper(flatten_truncate_function, max_length=510, input_column="context", output_column='flat_context')
    truncation_name = "Uniform MILP" if config.uniform_truncation else "Max Average"

    flattened_ds = flatten_contexts_in_dataset(targets_contexts_ds,
                                               output_basedir_path,
                                               flatten_truncate_batch_map_func,
                                               truncation_type=truncation_name,
                                               num_proc=config.threads,
                                               keep_columns=None,
                                               save_dataset=True)
            


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dataset preparation")
    parser.add_argument("--remove-target-percentage", default=0.9, type=float)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--seed", default=43, type=int)
    parser.add_argument("--output-basedir", default=None, type=str)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument('--uniform-truncation', default=False, action='store_true', help="Use MILP uniform truncation")

    config = parser.parse_args()
    prepare_dataset_from_chunks(config)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time taken: {end - start}s')
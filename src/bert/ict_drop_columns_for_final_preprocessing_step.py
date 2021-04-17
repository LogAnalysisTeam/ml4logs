from pathlib import Path
from typing import List
import time
from datasets import Value, Sequence, Features

import logging
import os

from dataset_utils import my_caching_load_from_disk

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

log = logging.getLogger(__name__)

def prepare_dataset_from_chunks(config):
    assert config.dataset is not None, "There must be a log file"
    assert config.output_basedir is not None
    dataset_path = Path(config.dataset)
    output_basedir_path = Path(config.output_basedir)
    NAME = f'Dropping columns and consolidating into single file {dataset_path} to basedir {output_basedir_path}'
    log.info(NAME)
    log.info(config)

    ds = my_caching_load_from_disk(dataset_path)

    output_dir = output_basedir_path / dataset_path.stem
    log.info(f'Mapping and saving to {output_dir}')
    flat_dropped_ds = ds.map(function=None,
                             batched=True,
                             batch_size=10000,
                             writer_batch_size=10000,
                             remove_columns=['context', 'context_text', 'target_text'],
                             features=Features({'target': Sequence(Value('int32')), 'flat_context': Sequence(Value('int32'))}))
    log.info(f'Saving to {output_dir}')
    flat_dropped_ds.save_to_disk(output_dir)
            


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Consolidating dataset into single file with only the training columns")
    parser.add_argument("--output-basedir", default=None, type=str)
    parser.add_argument("--dataset", default=None, type=str)

    config = parser.parse_args()
    prepare_dataset_from_chunks(config)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time taken: {end - start}s')
import fasttext
import time
from pathlib import Path
from typing import List, Union, Dict, Optional
from collections import defaultdict
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
import re

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

log = logging.getLogger(__name__)

def load_hdfs1_log_file_grouped(log_path: Path) -> Dict:
    HDFS1_BLK_ID_PATTERN = re.compile(r'(blk_-?\d+)')
    blocks = defaultdict(list)
    with log_path.open(mode='r') as f:
        for line in f:
            block_id = HDFS1_BLK_ID_PATTERN.search(line).group()
            blocks[block_id].append(line.strip())
    return blocks

def train_fasttext_model(config):
    assert config.model_path is not None, "Model path must be specified!"
    assert config.input_dataset_log_file is not None, "Input dataset must be specified!"
    assert config.input_dataset_labels_csv is not None, "Input dataset labels must be specified!"
    assert config.output_parent_folder is not None, "Output parent folder must be specified!"

    dataset_path = Path(config.input_dataset_log_file)
    label_path = Path(config.input_dataset_labels_csv)
    dataset_name = f'{dataset_path.stem}/labeled_embedding_from_{Path(config.model_path).stem}'
    output_path = Path(config.output_parent_folder) / dataset_name

    assert (not output_path.exists()) or (output_path.exists() and output_path.is_dir() and not any(output_path.iterdir())), f"Output path {output_path} is not empty, can't create dataset"
    log.info(dataset_name)
    log.info(config)

    log.info(f"Loading dataset from {dataset_path}")
    start = time.time()
    dataset = load_hdfs1_log_file_grouped(dataset_path)
    labels_df = pd.read_csv(label_path, converters={'Label': lambda x: x == 'Anomaly'})
    log.info(f'Done, time taken: {time.time() - start}s')

    log.info(f"Embedding dataset with model from {config.model_path}")
    start = time.time()

    model = fasttext.load_model(config.model_path)

    embedded_dataset = {block_id: np.array([model.get_sentence_vector(line.strip()) for line in block_lines]) for block_id, block_lines in tqdm(dataset.items())}
    log.info(f'Done, time taken: {time.time() - start}s')
    log.info(f"Saving dataset to {output_path}")
    start = time.time()
    ordered_by_label = [embedded_dataset[block_id] for block_id in labels_df['BlockId']]
    output_path.mkdir(parents=True, exist_ok=True)

    X_path = output_path / 'X.pickle'
    y_path = output_path / 'y.npy'
    with X_path.open('wb') as f:
        pickle.dump(ordered_by_label, f)
    np.save(y_path, labels_df['Label'].to_numpy(dtype=np.int8))

    log.info(f'Done, time taken: {time.time() - start}s')

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train fasttext model")
    parser.add_argument("--output-parent-folder", default=None, type=str)
    parser.add_argument('--model-path', default=None, type=str, help="Dataset to train from")
    parser.add_argument("--input-dataset-log-file", default=None, type=str)
    parser.add_argument("--input-dataset-labels-csv", default=None, type=str)

    config = parser.parse_args()
    train_fasttext_model(config)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time taken: {end - start}s')
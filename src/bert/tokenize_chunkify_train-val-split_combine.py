from datasets import load_dataset, load_from_disk, concatenate_datasets
from pathlib import Path
from typing import List
import numpy as np
import time
import subprocess
import json
import sys
import logging
import os

from dataset_pipeline import tokenize_chunkify_split_pipeline_single_dataset, combine_datasets


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

log = logging.getLogger(__name__)


def tokenize_chunkify_split_combine_pipeline(config):
    assert config.output_dir is not None, "There must be an output directory"

    output_dir_path = Path(config.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    used_dataset_str_paths = {
        'train-data-HDFS1-cv1-1-time-ordered': '/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/no_timestamps_train-data-HDFS1-cv1-1-time-ordered.log',
        'HDFS2-secondarynamenode': '/home/cernypro/dev/source/ml4logs/data/interim/HDFS2/no_timestamps_cleaned/hadoop-hdfs-secondarynamenode-mesos-01.log',
        'HDFS2-namenode': '/home/cernypro/dev/source/ml4logs/data/interim/HDFS2/no_timestamps_cleaned/hadoop-hdfs-namenode-mesos-01.log',
        'HDFS2-datanode-01': '/home/cernypro/dev/source/ml4logs/data/interim/HDFS2/no_timestamps_cleaned/hadoop-hdfs-datanode-mesos-01.log',
        'HDFS2-datanode-13': '/home/cernypro/dev/source/ml4logs/data/interim/HDFS2/no_timestamps_cleaned/hadoop-hdfs-datanode-mesos-13.log',
        'Spark': '/home/cernypro/dev/source/ml4logs/data/interim/Spark/no_timestamps_spark.log',
        'Zookeeper': '/home/cernypro/dev/source/ml4logs/data/interim/Zookeeper/no_timestamps_zookeeper.log',
        'BGL': '/home/cernypro/dev/source/ml4logs/data/interim/BGL/no_timestamps_bgl.log',
        'Hadoop': '/home/cernypro/dev/source/ml4logs/data/interim/Hadoop/no_timestamps_hadoop.log',
    }

    desired_contexts_to_take = {
        'train-data-HDFS1-cv1-1-time-ordered': 60000,
        'HDFS2-secondarynamenode': 60000,
        'HDFS2-namenode': 60000,
        'HDFS2-datanode-01': 60000,
        'HDFS2-datanode-13': 60000,
        'Spark': 240000,
        'Zookeeper': 7432,
        'BGL': 60000,
        'Hadoop': 39336,
    }

    assert set(desired_contexts_to_take.keys()) == set(used_dataset_str_paths.keys()), "Dataset dictionaries with paths and desired context counts must have same keys"
    log.info(config)

    used_dataset_paths = {name: Path(str_path) for name, str_path in used_dataset_str_paths.items()}

    log.info(f'Loading datasets')
    start = time.time()
    datasets = {name: load_dataset('text', data_files=str(path), split='train') for name, path in used_dataset_paths.items()}
    log.info(f"Loading datasets Time taken: {time.time() - start}s")

    train_val_splits_datasets = {name: tokenize_chunkify_split_pipeline_single_dataset(ds,
                                                                                       used_dataset_paths[name].parent / used_dataset_paths[name].stem,
                                                                                       tokenizer_bert_model=config.bert_model,
                                                                                       chunk_size=config.context_sentence_count, 
                                                                                       desired_total_chunks_taken=desired_contexts_to_take[name],
                                                                                       val_ratio=config.val_ratio,
                                                                                       seed=config.seed,
                                                                                       drop_last_incomplete_chunk=True,
                                                                                       save_dataset=True, 
                                                                                       ds_name_for_logging=name)
                                for name, ds in datasets.items()}

    train_datasets = {name: ds_dict['train'] for name, ds_dict in train_val_splits_datasets.items()}
    val_datasets = {name: ds_dict['val'] for name, ds_dict in train_val_splits_datasets.items()}

    combined_base_path = output_dir_path / f'shuffled_seed-{config.seed}'
    train_path = combined_base_path / 'train'
    val_path = combined_base_path / 'val'

    log.info(f'Combining train datasets to {train_path}')
    combined_train_ds = combine_datasets(train_datasets, output_path=train_path, seed=config.seed, save_dataset=True)
    
    log.info(f'Combining validation datasets to {val_path}')
    combined_train_ds = combine_datasets(val_datasets, output_path=val_path, seed=config.seed, save_dataset=True)

    split_sizes = {name: {'train': len(train_datasets[name]), 'val': len(val_datasets[name])} for name in used_dataset_str_paths.keys()}
    actual_contexts_counts = {name: sum(split.values()) for name, split in split_sizes.items()}


    INFO_DICT = {
        'datasets': list(used_dataset_str_paths.keys()),
        'seed': config.seed,
        'val_ratio': config.val_ratio,
        'log_paths': used_dataset_str_paths,
        'desired_context_counts': desired_contexts_to_take,
        'actual_context_counts': actual_contexts_counts,
        'split_sizes_in_contexts' : split_sizes,
    }

    log.info(json.dumps(INFO_DICT, indent=4))

    INFO_FILE_PATH = output_dir_path / "combining_info.json"

    INFO_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with INFO_FILE_PATH.open(mode='w') as f:
        f.write(json.dumps(INFO_DICT, indent=4))

            


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dataset transformation pipeline to create combined train-val dataset")
    parser.add_argument('--seed', default=43, type=int)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument('--bert-model', default="distilbert-base-cased", type=str, help="Pretrained Transformer for tokenizer.")
    parser.add_argument("--context-sentence-count", default=10, type=int)

    config = parser.parse_args()
    tokenize_chunkify_split_combine_pipeline(config)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    log.info(f'Total time taken: {end - start}s')
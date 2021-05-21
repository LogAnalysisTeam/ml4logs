#
# Original work at: https://github.com/LogAnalysisTeam/methods4logfiles/
#
# Modified for the purposes of Contextual Embeddings for Anomaly Detection in Logs thesis
#
#

import pandas as pd
import numpy as np
import os
from collections import defaultdict
import re
from typing import Dict, Union, Generator, DefaultDict
from sklearn import model_selection

from logparser import parse_file_drain3, save_drain3_to_file

SEED = 160121


def load_labels(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, converters={'Label': lambda x: True if x == 'Anomaly' else False})
    return df


def load_data(file_path: str) -> DefaultDict:
    traces = defaultdict(list)

    regex = re.compile(r'(blk_-?\d+)')  # pattern eg. blk_-1608999687919862906

    with open(file_path, 'r') as f:
        for line in f:
            block_id = find_block_id_in_log(regex, line)
            traces[block_id].append(line)
    return traces


def find_block_id_in_log(regex: re.Pattern, line: str) -> str:
    res = regex.search(line)
    return res.group()


def save_logs_to_file(data: Dict, file_path: str):
    with open(file_path, 'w') as f:
        for logs in data.values():
            f.writelines(logs)


def save_labels_to_file(data: pd.DataFrame, file_path: str):
    data.replace({True: 'Anomaly', False: 'Normal'}).to_csv(file_path, index=False)


def get_data_by_indices(data: Union[DefaultDict, Dict], labels: pd.DataFrame) -> Dict:
    ret = {block_id: data[block_id] for block_id in labels['BlockId']}
    return ret


def stratified_train_test_split(data: Union[DefaultDict, Dict], labels: pd.DataFrame, test_size: float,
                                seed: int) -> tuple:
    # assumes that one block is one label, otherwise it would generate more data
    train_labels, test_labels = model_selection.train_test_split(labels, stratify=labels['Label'], test_size=test_size,
                                                                 random_state=seed)

    train_data = get_data_by_indices(data, train_labels)
    test_data = get_data_by_indices(data, test_labels)
    return train_data, test_data, train_labels, test_labels


def process_raw_hdfs(data_dir: str, output_dir: str = None, save_to_file: bool = True, test_size: float = 0.1) -> tuple:
    """
    The logs are sliced into traces according to block ids. Then each trace associated with a specific block id is
    assigned a ground-truth label.
    :return:
    """
    labels = load_labels(os.path.join(data_dir, 'anomaly_label.csv'))
    data = load_data(os.path.join(data_dir, 'HDFS.log'))
    data_drain3 = parse_file_drain3(data)

    train_data, test_data, train_labels, test_labels = stratified_train_test_split(data, labels, seed=SEED,
                                                                                   test_size=test_size)
    train_data_drain3 = get_data_by_indices(data_drain3, train_labels)
    test_data_drain3 = get_data_by_indices(data_drain3, test_labels)

    if save_to_file and output_dir:
        save_logs_to_file(train_data, os.path.join(output_dir, 'train-data-HDFS1.log'))
        save_logs_to_file(test_data, os.path.join(output_dir, 'test-data-HDFS1.log'))
        save_drain3_to_file(train_data_drain3, os.path.join(output_dir, 'train-data-Drain3-HDFS1.log'))
        save_drain3_to_file(test_data_drain3, os.path.join(output_dir, 'test-data-Drain3-HDFS1.log'))
        save_labels_to_file(train_labels, os.path.join(output_dir, 'train-labels-HDFS1.csv'))
        save_labels_to_file(test_labels, os.path.join(output_dir, 'test-labels-HDFS1.csv'))
    return (train_data, test_data, train_labels, test_labels), (train_data_drain3,)


def get_train_val_hdfs(data: Dict, labels: pd.DataFrame, n_folds: int, test_size: float = 0.1) -> Generator:
    if n_folds == 1:  # it isn't CV but train_test_split
        yield stratified_train_test_split(data, labels, seed=SEED, test_size=test_size)
    else:
        skf = model_selection.StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
        for train_index, test_index in skf.split(np.zeros(len(labels)), labels['Label']):  # data is not important here
            train_labels = labels.iloc[train_index]
            test_labels = labels.iloc[test_index]
            train_data = get_data_by_indices(data, train_labels)
            test_data = get_data_by_indices(data, test_labels)
            yield train_data, test_data, train_labels, test_labels


def prepare_and_save_splits(data_dir: str, output_dir: str, n_folds: int):
    (train_data_logs, _, train_labels_logs, _), (train_data_drain3,) = process_raw_hdfs(data_dir, output_dir)
    splits = get_train_val_hdfs(train_data_logs, train_labels_logs, n_folds)
    for idx, (train_data, test_data, train_labels, test_labels) in enumerate(splits, start=1):
        save_logs_to_file(train_data, os.path.join(output_dir, f'train-data-HDFS1-cv{idx}-{n_folds}.log'))
        save_logs_to_file(test_data, os.path.join(output_dir, f'val-data-HDFS1-cv{idx}-{n_folds}.log'))
        save_labels_to_file(train_labels, os.path.join(output_dir, f'train-labels-HDFS1-cv{idx}-{n_folds}.csv'))
        save_labels_to_file(test_labels, os.path.join(output_dir, f'val-labels-HDFS1-cv{idx}-{n_folds}.csv'))

    # save data for baseline methods parsed by Drain3
    splits = get_train_val_hdfs(train_data_drain3, train_labels_logs, n_folds)
    for idx, (train_data, test_data, train_labels, test_labels) in enumerate(splits, start=1):
        save_drain3_to_file(train_data, os.path.join(output_dir, f'train-data-Drain3-HDFS1-cv{idx}-{n_folds}.binlog'))
        save_drain3_to_file(test_data, os.path.join(output_dir, f'val-data-Drain3-HDFS1-cv{idx}-{n_folds}.binlog'))

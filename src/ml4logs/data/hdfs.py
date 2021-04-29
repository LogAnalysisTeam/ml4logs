
# ===== IMPORTS =====
# === Standard library ===
from collections import defaultdict, OrderedDict
import datetime
import logging
import os
import pathlib
import re
import typing
from typing import Dict, Union, Generator

# === Thirdparty ===
import numpy as np
import pandas as pd
from sklearn import model_selection

# === Local ===
import ml4logs

# ===== GLOBALS =====
logger = logging.getLogger(__name__)

# ===== FUNCTIONS =====

# Based on Martin Korytak: https://github.com/LogAnalysisTeam/methods4logfiles

class HDFSImporter:
    def __init__(self, data_dir: str, output_dir: str, n_folds: int = 1, test_size: float = 0.1, val_size: float = 0.1, SEED: int = 160121):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.test_size = test_size
        self.val_size = val_size
        self.SEED = SEED

    def prepare_and_save_splits(self):
        (train_data_logs, _, train_labels_logs, _) = self.process_raw_hdfs(test_size=self.test_size)
        splits = self.get_train_val_hdfs(
            train_data_logs, train_labels_logs, val_size=self.val_size)
        for idx, (train_data, test_data, train_labels, test_labels) in enumerate(splits, start=1):
            self.save_logs_to_file(
                train_data, f'train/cv{idx}-{self.n_folds}/data.log')
            self.save_logs_to_file(
                test_data, f'val/cv{idx}-{self.n_folds}/data.log')
            self.save_labels_to_file(
                train_labels, f'train/cv{idx}-{self.n_folds}/labels.csv')
            self.save_labels_to_file(
                test_labels, f'val/cv{idx}-{self.n_folds}/labels.csv')


    def process_raw_hdfs(self, save_to_file: bool = True, test_size: float = 0.1) -> tuple:
        """
        The logs are sliced into traces according to block ids. Then each trace associated with a specific block id is
        assigned a ground-truth label.
        :return:
        """
        labels = self.load_labels('anomaly_label.csv')
        data = self.load_data('HDFS.log')
        # remove labels for blocks not in data (this might happen if using only subset of data for debugging, e.g., HDFS1_100k) 
        labels = labels[labels["BlockId"].isin(data.keys())]

        train_data, test_data, train_labels, test_labels = _stratified_train_test_split(data, labels, seed=self.SEED,
                                                                                        test_size=test_size)

        if save_to_file and self.output_dir:
            self.save_logs_to_file(train_data, 'train/data.log')
            self.save_logs_to_file(test_data, 'test/data.log')
            self.save_labels_to_file(train_labels, 'train/labels.csv')
            self.save_labels_to_file(test_labels, 'test/labels.csv')
        return (train_data, test_data, train_labels, test_labels)

    def get_train_val_hdfs(self, data: Dict, labels: pd.DataFrame, val_size: float = 0.1) -> Generator:
        if self.n_folds == 1:  # it isn't CV but train_test_split
            yield _stratified_train_test_split(data, labels, seed=self.SEED, test_size=val_size)
        else:
            skf = model_selection.StratifiedKFold(
                self.n_folds, shuffle=True, random_state=self.SEED)
            # data is not important here
            for train_index, test_index in skf.split(np.zeros(len(labels)), labels['Label']):
                train_labels = labels.iloc[train_index]
                test_labels = labels.iloc[test_index]
                train_data = _get_data_by_indices(data, train_labels)
                test_data = _get_data_by_indices(data, test_labels)
                yield train_data, test_data, train_labels, test_labels

    def load_labels(self, file_name: str) -> pd.DataFrame:
        return load_labels(pathlib.Path(self.data_dir, file_name))


    def load_data(self, file_name: str) -> typing.OrderedDict:
        return load_data(pathlib.Path(self.data_dir, file_name))


    def save_logs_to_file(self, data: Dict, file_name: str):
        file_path = pathlib.Path(self.output_dir, file_name)
        ml4logs.utils.mkdirs(files=[file_path])
        with open(file_path, 'w') as f:
            for logs in data.values():
                f.writelines(logs)

    def save_labels_to_file(self, data: pd.DataFrame, file_name: str):
        file_path = pathlib.Path(self.output_dir, file_name)
        ml4logs.utils.mkdirs(files=[file_path])
        data.replace({True: 'Anomaly', False: 'Normal'}).to_csv(
            file_path, index=False)


# --- HELPER FUNCIONS

def load_labels(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, converters={
                        'Label': lambda x: True if x == 'Anomaly' else False})
    return df

def load_data(file_path: str) -> typing.OrderedDict:
    traces = OrderedDict()

    # pattern eg. blk_-1608999687919862906
    regex = re.compile(r'(blk_-?\d+)')

    with open(file_path, 'r') as f:
        for line in f:
            block_id = _find_block_id_in_log(regex, line)
            tlst = traces.get(block_id, [])
            tlst.append(line)
            traces[block_id] = tlst
    return traces

def _find_block_id_in_log(regex: re.Pattern, line: str) -> str:
    res = regex.search(line)
    return res.group()


def _get_data_by_indices(data: Union[typing.OrderedDict, Dict], labels: pd.DataFrame) -> Dict:
    ret = {block_id: data[block_id] for block_id in labels['BlockId'] if block_id in data}
    return ret


def _stratified_train_test_split(data: Union[typing.OrderedDict, Dict], labels: pd.DataFrame, test_size: float,
                                 seed: int) -> tuple:
    # assumes that one block is one label, otherwise it would generate more data
    train_labels, test_labels = model_selection.train_test_split(labels, stratify=labels['Label'], test_size=test_size,
                                                                 random_state=seed)

    train_data = _get_data_by_indices(data, train_labels)
    test_data = _get_data_by_indices(data, test_labels)
    return train_data, test_data, train_labels, test_labels

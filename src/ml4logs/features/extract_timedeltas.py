# ===== IMPORTS =====
# === Standard library ===
from collections import defaultdict, OrderedDict
from datetime import datetime
import logging
import pathlib
from pathlib import Path
import re
import typing
from typing import Generator, Iterable, List, Dict

# === Thirdparty ===
import joblib
import numpy as np
import pandas as pd

# === Local ===
import ml4logs
from ml4logs.data.hdfs import load_data_as_dict

# based on https://github.com/LogAnalysisTeam/methods4logfiles/blob/main/src/features/hdfs.py

# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def extract_timedeltas(args):
    data_dir = Path(args['data_dir'])
    ml4logs.utils.mkdirs(folders=[data_dir])

    logger.info('Starting time delta extraction')
    for pair in args["pairs"]:
        logs_path = Path(data_dir, pair["logs_name"])
        timedeltas_path = Path(data_dir, pair["timedeltas_name"])
        logger.info(f'Processing: {logs_path}')

        data = load_data_as_dict(logs_path)
        tdeltas = []
        nlines = 0
        for block_id, log_lines in data.items():
            block_tdeltas = get_timedeltas(log_lines)
            nlines += len(log_lines)
            tdeltas.append(block_tdeltas)
        tdeltas = np.concatenate(tdeltas)
        assert len(tdeltas) == nlines

        logger.info(f'Processed {len(data)} blocks,  {len(tdeltas)} log lines')

        logger.info(f'Saving dataset as: {timedeltas_path}')
        np.save(timedeltas_path, tdeltas)

# ===== HELPER FUNCTIONS =====
def search(regex: re.Pattern, line: str) -> str:
    res = regex.search(line)
    if not res:  # temporal check!!
        raise AssertionError('Nothing found!!!!!')
    return res.group(1)


def get_datetime(timestamp: str) -> datetime:
    datetime_object = datetime.strptime(timestamp, '%y%m%d %H%M%S')
    return datetime_object


def to_seconds(timedelta: np.array) -> np.array:
    return np.vectorize(lambda x: int(x.total_seconds()))(timedelta)


def calculate_timedeltas_from_timestamps(timestamps: np.array) -> np.array:
    # timedeltas = np.zeros(shape=timestamps.shape, dtype=np.int32)
    # timedeltas[1:] = to_seconds(timestamps[1:] - timestamps[:-1])
    # timedeltas[timedeltas == 0] = 1  # due to undefined behaviour of log10
    # # timedeltas += 1 # we don't lose the information about difference 1
    # timedeltas = np.log10(timedeltas)  # decrease importance of large time differences
    # return timedeltas
    timedeltas = np.ones(shape=timestamps.shape, dtype=np.int32)  # init as 1 since log10(0) is undefined
    if len(timedeltas) > 1: # filter out rare case of one log line per block 
        timedeltas[1:] += to_seconds(timestamps[1:] - timestamps[:-1])  # we don't lose the information if the delta is 1
    timedeltas = np.log10(timedeltas)  # decrease importance of large time differences
    return timedeltas


def get_timedeltas(block_of_logs: List) -> np.array:
    datetime_from_line = re.compile(r'(^\d{6} \d{6})')

    timestamps = np.empty(shape=(len(block_of_logs),), dtype=np.object)
    for i, log in enumerate(block_of_logs):
        str_timestamp = search(datetime_from_line, log)
        timestamps[i] = get_datetime(str_timestamp)

    timedeltas = calculate_timedeltas_from_timestamps(timestamps)
    return timedeltas
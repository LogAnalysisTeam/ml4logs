# ===== IMPORTS =====
# === Standard library ===
import datetime
import logging
import pathlib
import re

# === Thirdparty ===
import numpy as np
import pandas as pd

# === Local ===
import ml4logs


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def prepare(args):
    HANDLERS = {
        'HDFS1': prepare_hdfs_1,
        'HDFS2': prepare_hdfs_2,
        'BGL': prepare_bgl,
        'Thunderbird': prepare_thunderbird
    }

    HANDLERS[args['dataset']](args)


def prepare_hdfs_1(args):
    in_dir = pathlib.Path(args['in_dir'])
    logs_path = pathlib.Path(args['logs_path'])
    blocks_path = pathlib.Path(args['blocks_path'])
    labels_path = pathlib.Path(args['labels_path'])
    timedeltas_path = pathlib.Path(args['timedeltas_path'])

    ml4logs.utils.mkdirs(files=[logs_path, labels_path,
                                blocks_path, timedeltas_path])

    in_labels_path = in_dir / 'anomaly_label.csv'
    logger.info('Read labels from \'%s\'', in_labels_path)
    labels_df = pd.read_csv(in_labels_path)
    logger.info('Map labels to 0/1 (normal/anomaly)')
    labels = labels_df['Label'].map({'Normal': 0, 'Anomaly': 1}).to_numpy()
    logger.info('Save mapped labels into \'%s\'', labels_path)
    np.save(labels_path, labels)

    in_logs_path = in_dir / 'HDFS.log'
    logger.info('Move \'%s\' to \'%s\'', in_logs_path, logs_path)
    in_logs_path.replace(logs_path)

    blk_mapping = dict(zip(labels_df['BlockId'], labels_df.index))
    pattern = re.compile(r'(blk_-?\d+)')
    blocks = []
    timestamps = []
    logger.info('Extract blocks and timestamps from log lines')
    with logs_path.open() as logs_in_f:
        for line in logs_in_f:
            match = pattern.search(line.strip())
            blocks.append(blk_mapping[match.group()])
            dt_str = line[:13]
            dt = datetime.datetime.strptime(dt_str, r'%y%m%d %H%M%S')
            timestamps.append(dt.timestamp())
    blocks = np.array(blocks)
    timestamps = np.array(timestamps)

    logger.info('Save blocks into \'%s\'', blocks_path)
    np.save(blocks_path, blocks)

    logger.info('Compute timedeltas')
    indexes = {}
    values = {}
    for idx, (block, array) in enumerate(zip(blocks, timestamps)):
        list_values = values.setdefault(block, list())
        list_values.append(array)
        list_idx = indexes.setdefault(block, list())
        list_idx.append(idx)
    values = {block: np.stack(arrays) for block, arrays in values.items()}
    indexes = {block: np.stack(arrays) for block, arrays in indexes.items()}
    timedeltas = np.zeros_like(timestamps)
    for block in np.unique(blocks):
        tds = np.zeros_like(values[block])
        tds[1:] = values[block][1:] - values[block][:-1]
        timedeltas[indexes[block]] = tds
    timedeltas = np.expand_dims(timedeltas, axis=timedeltas.ndim)

    logger.info('Save timedeltas into \'%s\'', timedeltas_path)
    np.save(timedeltas_path, timedeltas)


def prepare_hdfs_2(args):
    raise NotImplementedError('Prepare is not implemented for HDFS2 dataset')


def prepare_bgl(args):
    in_path = pathlib.Path(args['in_dir']) / 'BGL.log'
    logs_path = pathlib.Path(args['logs_path'])
    labels_path = pathlib.Path(args['labels_path'])

    ml4logs.utils.mkdirs(files=[logs_path, labels_path])

    n_lines = ml4logs.utils.count_file_lines(in_path)
    step = n_lines // 10
    logger.info('Start splitting labels and log messages')
    labels = []
    with in_path.open(encoding='utf8') as in_f, \
            logs_path.open('w', encoding='utf8') as logs_out_f:
        for i, line in enumerate(in_f):
            label, raw_log = tuple(line.strip().split(maxsplit=1))
            logs_out_f.write(f'{raw_log}\n')
            labels.append(0 if label == '-' else 1)
            if i % step <= 0:
                logger.info('Processed %d / %d lines', i, n_lines)
    logger.info('Save labels into \'%s\'', labels_path)
    np.save(labels_path, np.array(labels))


def prepare_thunderbird(args):
    in_path = pathlib.Path(args['in_dir']) / 'Thunderbird.log'
    logs_path = pathlib.Path(args['logs_path'])
    labels_path = pathlib.Path(args['labels_path'])

    ml4logs.utils.mkdirs(files=[logs_path, labels_path])

    n_lines = ml4logs.utils.count_file_lines(in_path)
    step = n_lines // 10
    logger.info('Start splitting labels and log messages')
    labels = []
    with in_path.open(encoding='utf8') as in_f, \
            logs_path.open('w', encoding='utf8') as logs_out_f:
        for i, line in enumerate(in_f):
            label, raw_log = tuple(line.strip().split(maxsplit=1))
            logs_out_f.write(f'{raw_log}\n')
            labels.append(0 if label == '-' else 1)
            if i % step <= 0:
                logger.info('Processed %d / %d lines', i, n_lines)
    logger.info('Save labels into \'%s\'', labels_path)
    np.save(labels_path, np.array(labels))

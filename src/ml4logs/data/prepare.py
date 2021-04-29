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
from ml4logs.data.hdfs import HDFSImporter

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
    out_dir = pathlib.Path(args['out_dir'])

    importer = HDFSImporter(in_dir, out_dir, n_folds=1, test_size=0.1, val_size=0.1, SEED=160121)
    importer.prepare_and_save_splits()


def prepare_hdfs_2(args):
    raise NotImplementedError('Prepare is not implemented for HDFS2 dataset')


def prepare_bgl(args):
    in_dir = pathlib.Path(args['in_dir'])
    split_labels(args, in_dir / 'BGL.log', '-')


def prepare_thunderbird(args):
    in_dir = pathlib.Path(args['in_dir'])
    split_labels(args, in_dir / 'Thunderbird.log', '-')


def split_labels(args, in_path, normal_label):
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
            labels.append(0 if label == normal_label else 1)
            if i % step <= 0:
                logger.info('Processed %d / %d lines', i, n_lines)
    logger.info('Save labels into \'%s\'', labels_path)
    np.save(labels_path, np.array(labels))

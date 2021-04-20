# ===== IMPORTS =====
# === Standard library ===
import logging
import pathlib

# === Thirdparty ===
import numpy as np

# === Local ===
import ml4logs


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def aggregate_by_blocks(args):
    features_path = pathlib.Path(args['features_path'])
    blocks_path = pathlib.Path(args['blocks_path'])
    labels_path = pathlib.Path(args['labels_path'])
    dataset_path = pathlib.Path(args['dataset_path'])

    ml4logs.utils.mkdirs(files=[dataset_path])

    logger.info('Load features, blocks and labels')
    features = np.load(features_path)
    blocks = np.load(blocks_path)
    labels = np.load(labels_path)

    logger.info('Group feature arrays by blocks')
    groups = {}
    for block, array in zip(blocks, features):
        list_ = groups.setdefault(block, list())
        list_.append(array)

    methods = {
        'sum': lambda a: np.sum(a, axis=0),
        'average': lambda a: np.average(a, axis=0),
        'max': lambda a: np.nanmax(a, axis=0),
        'min': lambda a: np.nanmin(a, axis=0),
        'count_vector': lambda a: np.bincount(a, minlength=features.max() + 1)
    }

    logger.info('Aggregate features using \'%s\' method', args['method'])
    Xs = []
    Ys = []
    for block, arrays in groups.items():
        Xs.append(methods[args['method']](np.stack(arrays)))
        Ys.append(labels[block])
    X = np.stack(Xs)
    Y = np.stack(Ys)

    logger.info('X = %s, Y = %s', X.shape, Y.shape)
    logger.info('Save dataset into \'%s\'', dataset_path)
    np.savez(dataset_path, X=X, Y=Y)


def aggregate_by_lines(args):
    features_path = pathlib.Path(args['features_path'])
    labels_path = pathlib.Path(args['labels_path'])
    dataset_path = pathlib.Path(args['dataset_path'])

    ml4logs.utils.mkdirs(files=[dataset_path])

    logger.info('Load features and labels')
    features = np.load(features_path)
    labels = np.load(labels_path)[:len(features)]

    logger.info('X = %s, Y = %s', features.shape, labels.shape)
    logger.info('Save dataset into \'%s\'', dataset_path)
    np.savez(dataset_path, X=features, Y=labels)

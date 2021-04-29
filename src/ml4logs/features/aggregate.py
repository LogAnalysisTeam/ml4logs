# ===== IMPORTS =====
# === Standard library ===
from collections import defaultdict, OrderedDict
import logging
import pathlib
from pathlib import Path
import typing

# === Thirdparty ===
import joblib
import numpy as np
import pandas as pd

# === Local ===
import ml4logs
from ml4logs.data.hdfs import load_labels
from ml4logs.features.count_features import CountFeatureExtractor
from ml4logs.features.utils import load_features_as_dict

# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====

def _check_groups_and_labels(groups, labels):
    # groups dict of (BlockId -> list of log lines) should match the order of labels
    assert len(groups) == len(labels)
    assert np.all(list(groups.keys()) == labels)

def aggregate_by_blocks(args):
    assert args["method"] in ["bow", "tf-idf", "max"]

    features_path = Path(args['features_path'])
    labels_path = Path(args['labels_path'])
    dataset_path = Path(args['dataset_path'])

    ml4logs.utils.mkdirs(files=[dataset_path])

    logger.info(f'Loading features grouped by blocks,\n labels_path: {labels_path}, features_path: {features_path}')
    groups = load_features_as_dict(labels_path, features_path)
    logger.info(f"Loaded {len(groups)} groups")
    
    if "load_transform_path" in args: # read a once fitted transform, e.g., on training data
        load_transform_path = Path(args['load_transform_path'])
        logger.info(f"Loading aggregation transform (CountFeatureExtractor) as {load_transform_path}")
        fe = joblib.load(load_transform_path)
        X = fe.transform(groups)

    else:
        if args["method"] in ["bow", "tf-idf"]:
            fe = CountFeatureExtractor(method=args["method"], preprocessing="mean")
            logger.info("Fitting CountFeatureExtractor for the groups")
            X = fe.fit_transform(groups)
            if "save_transform_path" in args:
                save_transform_path = Path(args['save_transform_path'])
                logger.info(f"Saving aggregation transform (CountFeatureExtractor) as {save_transform_path}")
                joblib.dump(fe, save_transform_path)
        else:
            METHODS = {
                "max": lambda block: block.max(axis=0)
            }
            method = METHODS[args["method"]]
            first = list(groups.keys())[0]
            X = np.empty((len(groups), groups[first].shape[1]), dtype=np.float32)
            for i, (block_id, block) in enumerate(groups.items()):
                assert block.shape[0] > 0, f"Zero size block for {block_id}!"
                X[i, :] = method(block)

    logger.info('Loading labels')
    labels = load_labels(labels_path)
    _check_groups_and_labels(groups, labels.BlockId)
    Y = labels.Label.values

    logger.info('X = %s, Y = %s', X.shape, Y.shape)
    logger.info('Saving dataset into \'%s\'', dataset_path)
    np.savez(dataset_path, X=X, Y=Y)


def aggregate_by_lines(args):
    features_path = Path(args['features_path'])
    labels_path = Path(args['labels_path'])
    dataset_path = Path(args['dataset_path'])

    ml4logs.utils.mkdirs(files=[dataset_path])

    logger.info('Load features and labels')
    features = np.load(features_path)
    labels = np.load(labels_path)[:len(features)]

    logger.info('X = %s, Y = %s', features.shape, labels.shape)
    logger.info('Save dataset into \'%s\'', dataset_path)
    np.savez(dataset_path, X=features, Y=labels)

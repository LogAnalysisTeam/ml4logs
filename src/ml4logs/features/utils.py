# ===== IMPORTS =====
# === Standard library ===
from collections import defaultdict, OrderedDict
import logging
import pathlib
import typing

# === Thirdparty ===
import joblib
import numpy as np
import pandas as pd

# === Local ===
import ml4logs
from ml4logs.data.hdfs import load_labels
from ml4logs.features.count_features import CountFeatureExtractor

# ===== GLOBALS =====
logger = logging.getLogger(__name__)

def load_features(features_path: str) -> np.ndarray:
    data = np.load(features_path)
    assert 0 < data.ndim <= 2
    if data.ndim == 1:
        return data.reshape(-1, 1)
    return data

    
def load_features_as_dict(labels_path: str, features_path: str) -> typing.OrderedDict:
    features = load_features(features_path) # the order of features is same as the order of data
    labels = load_labels(labels_path)

    groups = OrderedDict()
    for row in labels.itertuples():
        off = row.BlockOffset
        groups[row.BlockId] = features[off:off + row.BlockSize]
    return groups

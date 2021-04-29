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
from ml4logs.data.hdfs import load_data, load_labels
from ml4logs.features.count_features import CountFeatureExtractor

# ===== GLOBALS =====
logger = logging.getLogger(__name__)

def load_features(features_path: str) -> np.ndarray:
    data = np.load(features_path)
    assert 0 < data.ndim <= 2
    if data.ndim == 1:
        return data.reshape(-1, 1)
    return data

    
def load_features_as_dict(data_path: str, features_path: str) -> typing.OrderedDict:
    data = load_data(data_path) # BlockId -> list of log lines OrderedDict, the block order corresponds to the order of features and labels
    features = load_features(features_path) # the order of features is same as the order of data
    # features = np.load(features_path) # the order of features is same as the order of data

    groups = OrderedDict()
        
    off = 0
    for block_id, log_lines in data.items():
        groups[block_id] = features[off:off+len(log_lines)]
        off += len(log_lines)
        
    return groups

# ===== IMPORTS =====
# === Standard library ===
from collections import defaultdict, OrderedDict
import logging
import pathlib
import typing
from typing import Dict, Union, Generator

# === Thirdparty ===
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import tqdm_logging_wrapper

# === Local ===
import ml4logs

# ===== GLOBALS =====
logger = logging.getLogger(__name__)

def classify(Y: np.array, threshold: float) -> np.array:
    # classify based on threshold
    # from https://github.com/LogAnalysisTeam/methods4logfiles/blob/main/src/models/utils.py
    ret = np.zeros(shape=Y.shape)
    ret[Y >= threshold] = 1
    return ret

def get_metrics(T: np.array, Y: np.array, **kwargs) -> Dict:
    # based on https://github.com/LogAnalysisTeam/methods4logfiles/blob/main/src/models/utils.py
    # both T and Y should be classifications (0/1) 
    # returns JSON-serializable object
    results = OrderedDict()
    precision, recall, f1, _ = precision_recall_fscore_support(
        T, Y, average='binary', zero_division=0, **kwargs)
    results["precision"] = float(precision)
    results["recall"] = float(recall)
    results["f1"] = float(f1)
    results["mcc"] = float(matthews_corrcoef(T, Y, **kwargs))

    return results


def get_threshold_metrics(T: np.array, Y: np.array, **kwargs) -> Dict:
    # based on https://github.com/LogAnalysisTeam/methods4logfiles/blob/main/src/models/utils.py
    # T - target classifications (0/1)
    # Y - predicted scores (continuous
    # returns JSON-serializable object
    results = OrderedDict()
    auc = roc_auc_score(T, Y)
    ap = average_precision_score(T, Y)
    results["auc"] = float(auc)
    results["ap"] = float(ap)

    return results

def f1_score_binary(T: np.array, Y: np.array) -> np.float32:
    '''Calculate F1 score. Should be much faster than Sklearn one.
    
    The original implementation is written by Michal Haltuf on Kaggle. This one based on Shlomi Shmuel.
    
    Reference
    ---------
    - https://stackoverflow.com/a/62524723
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert T.shape == Y.shape
    
    tp = (T * Y).sum()
    tn = ((1 - T) * (1 - Y)).sum()
    fp = ((1 - T) * Y).sum()
    fn = (T * (1 - Y)).sum()
    
    tp_fp = tp + fp
    tp_fn = tp + fn

    # check if all predictions are 0
    precision = 0.0 if np.isclose(tp_fp, 0.0) else tp / tp_fp

    # check if all targets are 0
    assert not np.isclose(tp_fn, 0.0), "All targets are 0!"
    recall = tp / tp_fn
    
    p_r = precision + recall

    if np.isclose(p_r, 0.0):
        return 0.0
    else:
        return (precision*recall) / (precision + recall)


def find_optimal_threshold(T: np.array, Y: np.array) -> tuple:
    # from https://github.com/LogAnalysisTeam/methods4logfiles/blob/main/src/models/utils.py
    # T - target classifications (0/1)
    # Y - predicted scores

    ret = {}
    # logger.info(f"Y {type(Y)} {Y.shape} = {Y}")
    # logger.info(f"T {type(T)} {T.shape} = {T}")
    thresholds = set(Y[T == 1])
    logger.debug(f"# thresholds to test: {len(thresholds)}, T.shape = {T.shape}, Y.shape = {Y.shape}")
    thresholds_iter = tqdm(thresholds)
    with tqdm_logging_wrapper.wrap_logging_for_tqdm(thresholds_iter), thresholds_iter:
        for th in thresholds_iter:
            Ycls = classify(Y, th)

            # f1 = get_metrics(T, Ycls)[metric] # slow!
            f1 = f1_score_binary(T, Ycls) # custom implementation needed
            ret[th] = f1
    return max(ret.items(), key=lambda x: x[1])
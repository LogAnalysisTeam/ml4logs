#
# Original work at: https://github.com/LogAnalysisTeam/methods4logfiles/blob/main/src/models/metrics.py
#
#
#
#

from sklearn import metrics
import numpy as np
from typing import Dict


def precision(y_true: np.array, y_pred: np.array, **kwargs) -> float:
    # range(0, 1)
    return metrics.precision_score(y_true, y_pred, **kwargs)


def recall(y_true: np.array, y_pred: np.array, **kwargs) -> float:
    # range(0, 1)
    return metrics.recall_score(y_true, y_pred, **kwargs)


def f1_score(y_true: np.array, y_pred: np.array, **kwargs) -> float:
    # range(0, 1)
    return metrics.f1_score(y_true, y_pred, **kwargs)


def mcc_score(y_true: np.array, y_pred: np.array, **kwargs) -> float:
    # range(-1, 1)
    return metrics.matthews_corrcoef(y_true, y_pred, **kwargs)


def get_metrics(y_true: np.array, y_pred: np.array, **kwargs) -> Dict:
    # return JSON-serializable object
    return {
        'precision': float(precision(y_true, y_pred, **kwargs)),
        'recall': float(recall(y_true, y_pred, **kwargs)),
        'f1_score': float(f1_score(y_true, y_pred, **kwargs)),
        'mcc_score': float(mcc_score(y_true, y_pred, **kwargs))
    }


def metrics_report(y_true: np.array, y_pred: np.array, **kwargs):
    print('+----------------------------------+-----------+')
    print('| Metric                           | Score     |')
    print('+----------------------------------+-----------+')
    print(f'| Precision                        | {precision(y_true, y_pred, **kwargs):.5f}   |')
    print('+----------------------------------+-----------+')
    print(f'| Recall                           | {recall(y_true, y_pred, **kwargs):.5f}   |')
    print('+----------------------------------+-----------+')
    print(f'| F1 Score                         | {f1_score(y_true, y_pred, **kwargs):.5f}   |')
    print('+----------------------------------+-----------+')
    print(f'| Matthews Correlation Coefficient | {mcc_score(y_true, y_pred, **kwargs):.5f}   |')
    print('+----------------------------------+-----------+')


if __name__ == '__main__':
    np.random.seed(1)

    t = np.random.random(10)
    p = np.random.random(10)

    t = np.where(t > 0.5, 1, 0)
    p = np.where(p > 0.5, 1, 0)

    print('y_true:', t)
    print('y_pred:', p)

    print('precision:', precision(t, p))
    print('recall:', recall(t, p))
    print('F1 score:', f1_score(t, p))
    print('MCC score:', mcc_score(t, p))

    metrics_report(t, p)

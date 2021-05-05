# ===== IMPORTS =====
# === Standard library ===
import logging
import pathlib
import warnings
import json

# === Thirdparty ===
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from sklearn.metrics import (roc_auc_score,
                             average_precision_score,
                             precision_recall_fscore_support)

# === Local ===
import ml4logs
from ml4logs.models.utils import get_metrics, get_threshold_metrics

# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== CLASSES =====
class LinearSVCWrapper:
    def __init__(self, **kwargs):
        self._linear_svc = LinearSVC(**kwargs)
        self._clf = CalibratedClassifierCV(self._linear_svc)

    def fit(self, X, T):
        self._clf.fit(X, T)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

class IForestWrapper:
    def __init__(self, **kwargs):
        self._model = IForest(**kwargs)

    def fit(self, X, T):
        # unsupervised learning Targets not used
        self._model.fit(X)
        return self

    def predict(self, X):
        Y =  self._model.predict(X)
        return Y

    def predict_proba(self, X):
        probs = self._model.predict_proba(X)
        return probs

class IsolationForestWrapper:
    def __init__(self, **kwargs):
        self._model = IsolationForest(**kwargs)

    def fit(self, X, T):
        # unsupervised learning Targets not used
        self._model.fit(X)
        return self

    def predict(self, X):
        Y = (1 - (self._model.predict(X) + 1)/2).astype(np.int)
        return Y

    def predict_proba(self, X):
        # TODO: not realy continuous probabilities! Fix!
        # ps = self._model.score_samples(X).reshape(-1, 1)
        probs = np.zeros([X.shape[0], 2])
        Y = self.predict(X)
        probs[:, 0] = 1 - Y
        probs[:, 1] = Y
        return probs

class LOFWrapper:
    def __init__(self, **kwargs):
        self._model = LocalOutlierFactor(**kwargs)

    def fit(self, X, T):
        # LOF is not fit on training data, the fit is part of the prediction
        return self

    def predict(self, X):
        Y = (1 - (self._model.fit_predict(X) + 1)/2).astype(np.int)
        return Y

    def predict_proba(self, X):
        # TODO: not realy continuous probabilities! Fix!
        # ps = self._model.score_samples(X).reshape(-1, 1)
        probs = np.zeros([X.shape[0], 2])
        Y = self.predict(X)
        probs[:, 0] = 1 - Y
        probs[:, 1] = Y
        return probs

# ===== CONSTANTS =====
MODEL_CLASSES = {
    'logistic_regression': LogisticRegression,
    'decision_tree': DecisionTreeClassifier,
    'linear_svc': LinearSVCWrapper,
    'one_class_svm': OCSVM,

    # unsupervised
    'pca': PCA,
    'isolation_forest': IForestWrapper,
    'isolation_forest_sklearn': IsolationForestWrapper,
    'lof': LOF,
    'lof_sklearn': LOFWrapper,
}


# ===== FUNCTIONS =====
def train_test_models(args):
    train_path = pathlib.Path(args['train_path'])
    test_path = pathlib.Path(args['test_path'])
    stats_path = pathlib.Path(args['stats_path'])

    ml4logs.utils.mkdirs(files=[stats_path])

    train_npz = np.load(train_path)
    x_train, y_train = train_npz['X'], train_npz['Y']
    logger.info(f'Train data loaded from {train_path}, shapes X={x_train.shape} and Y={y_train.shape}')

    test_npz = np.load(test_path)
    x_test, y_test = test_npz['X'], test_npz['Y']
    logger.info(f'Test data loaded from {test_path}, shapes: X={x_test.shape} and Y={y_test.shape}')

    stats = {'step': args, 'metrics': {}}
    for m_dict in args['models']:
        logger.info('=== Using \'%s\' model ===', m_dict['name'])
        model = MODEL_CLASSES[m_dict['name']](**m_dict['args'])

        logger.info('Fitting train data to model')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(x_train, y_train)

        logger.info('Computing metrics on test data')
        c_pred = model.predict(x_test)
        y_pred = model.predict_proba(x_test)[:, 1]
        # logger.info(f"c_pred = {c_pred.shape}, y_pred = {y_pred.shape}, y_test = {y_test.shape}")

        metrics = get_metrics(y_test, c_pred)
        metrics.update(get_threshold_metrics(y_test, y_pred))

        logger.info(f'Precision = {metrics["precision"]:.2f}, Recall = {metrics["recall"]:.2f}, F1-score = {metrics["f1"]:.2f}')
        logger.info(f'MCC = {metrics["mcc"]:.2f}')
        logger.info(f'AUC = {metrics["auc"]:.2f}, AP = {metrics["ap"]:.2f}')

        stats['metrics'][m_dict['name']] = metrics

    logger.info('Saving metrics to \'%s\'', stats_path)
    stats_path.write_text(json.dumps(stats, indent=4))

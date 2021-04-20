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


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== CLASSES =====
class LinearSVCWrapper:
    def __init__(self, **kwargs):
        self._linear_svc = LinearSVC(**kwargs)
        self._clf = CalibratedClassifierCV(self._linear_svc)

    def fit(self, X, Y):
        self._clf.fit(X, Y)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)


# ===== CONSTANTS =====
MODEL_CLASSES = {
    'logistic_regression': LogisticRegression,
    'decision_tree': DecisionTreeClassifier,
    'linear_svc': LinearSVCWrapper,
    'lof': LOF,
    'one_class_svm': OCSVM,
    'isolation_forest': IForest,
    'pca': PCA
}


# ===== FUNCTIONS =====
def train_test_models(args):
    dataset_path = pathlib.Path(args['dataset_path'])
    stats_path = pathlib.Path(args['stats_path'])

    ml4logs.utils.mkdirs(files=[stats_path])

    logger.info('Load dataset from \'%s\'', dataset_path)
    with np.load(dataset_path) as dataset_npz:
        logger.info('Split with \'train size\' = %.2f', args['train_size'])
        x_train, x_test, y_train, y_test = train_test_split(
            dataset_npz['X'], dataset_npz['Y'],
            train_size=args['train_size'],
            stratify=dataset_npz['Y'],
            random_state=args['seed']
        )

    scaler = StandardScaler()
    logger.info('Scale train dataset using sklearn StandardScaler')
    x_train_scaled = scaler.fit_transform(x_train)
    logger.info('Scale test dataset using fitted sklearn StandardScaler')
    x_test_scaled = scaler.transform(x_test)

    stats = {'step': args, 'metrics': {}}
    for m_dict in args['models']:
        logger.info('=== Use \'%s\' model ===', m_dict['name'])
        model = MODEL_CLASSES[m_dict['name']](**m_dict['args'])

        logger.info('Fit train data to model')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(x_train_scaled, y_train)

        logger.info('Compute metrics on test data')
        c_pred = model.predict(x_test_scaled)
        y_pred = model.predict_proba(x_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, c_pred, average='binary', zero_division=0
        )
        logger.info('AUC = %.2f, AP = %.2f', auc, ap)
        logger.info('Precision = %.2f, Recall = %.2f, F1-score = %.2f',
                    precision, recall, f1)

        stats['metrics'][m_dict['name']] = {'auc': auc,
                                            'ap': ap,
                                            'precision': precision,
                                            'recall': recall,
                                            'f1': f1}

    logger.info('Save metrics into \'%s\'', stats_path)
    stats_path.write_text(json.dumps(stats, indent=4))

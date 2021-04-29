# ===== IMPORTS =====
# === Standard library ===
from collections import Counter
import logging
import typing
from typing import Dict, List

# === Thirdparty ===
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

# === Local ===


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====

# Based on https://github.com/LogAnalysisTeam/methods4logfiles/blob/main/src/features/feature_extractor.py

class CountFeatureExtractor(TransformerMixin):
    def __init__(self, method: str = None, preprocessing: str = None):
        assert method in ["bow", "tf-idf"]
        self.method = method
        self.feature_names = None
        self.preprocessing = preprocessing
        self._mu = None
        self._idf = None

    @staticmethod
    def _create_dataframe(data: Dict) -> pd.DataFrame:
        # expects a dictionary BlockId -> list of features (e.g., log template ids)
        # builds a DataFrame with one row per document and columns with feature counts, i.e., BOW representation
        occurrences = []
        for key, block in data.items():
#             occ_block = Counter(line[0] for line in block) # Martin's original
            occ_block = Counter(block.reshape(-1))
            occurrences.append(occ_block)

        df = pd.DataFrame(occurrences, index=data.keys())
        return df.fillna(0)

    def _add_missing_columns(self, data: pd.DataFrame):
        missing_columns = set(self.feature_names) - set(data.columns)
        for col in missing_columns:
            data[col] = 0

    def fit_transform(self, data: Dict, y=None, **fit_params) -> np.ndarray:
        dataframe = self._create_dataframe(data)

        self.feature_names = dataframe.columns

        ret = dataframe.to_numpy(dtype=np.float32)  # tf_{x_y} => the frequency of x in y document

        if self.method == 'tf-idf':
            df = np.sum(ret > 0, axis=0)  # the number of documents containing x
            self._idf = np.log(len(ret) / df).astype(np.float32)
            ret = ret * self._idf  # tf - idf

        if self.preprocessing == 'mean':
            self._mu = ret.mean(axis=0, dtype=np.float32)
            ret -= self._mu
        return ret

    def get_feature_names(self) -> List:
        return self.feature_names

    def transform(self, data: Dict):
        dataframe = self._create_dataframe(data)

        self._add_missing_columns(dataframe)
        ret = dataframe[self.feature_names].to_numpy(dtype=np.float32)

        if self.method == 'tf-idf':
            ret = ret * self._idf  # tf - idf

        if self.preprocessing == 'mean':
            ret -= self._mu
        return ret

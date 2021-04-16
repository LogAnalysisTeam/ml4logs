#
# Original work at: https://github.com/LogAnalysisTeam/methods4logfiles/blob/main/src/models/datasets.py
#
#
#
#

from __future__ import annotations

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, DefaultDict
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class EmbeddingDataset(Dataset):
    def __init__(self, data: np.ndarray, to: str = 'cpu', batch_size: int = 8):
        self.device = to
        self.batch_size = batch_size

        self.batches = self._prepare_data(data)

    @staticmethod
    def _get_occurrences(data: np.ndarray) -> DefaultDict:
        ret = defaultdict(list)
        for x in data:
            ret[x.shape].append(x)
        return ret

    def _prepare_data(self, data: np.ndarray) -> List:
        occurrences = self._get_occurrences(data)

        tensors = []
        for logs in occurrences.values():
            for i in range(0, len(logs), self.batch_size):
                batch = np.asarray(logs[i:i + self.batch_size])
                tensor = torch.from_numpy(batch).permute(0, 2, 1)  # transpose each example in the batch
                tensors.append(tensor.to(self.device))
        return tensors

    def __del__(self):
        del self.batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx) -> torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.batches[idx]


class CroppedDataset1D(Dataset):
    def __init__(self, data: np.ndarray, to: str = 'cpu', window: int = 25):
        self.device = to
        self.window = window

        self.tensor = self._prepare_data(data)

    def _prepare_data(self, data: np.ndarray) -> torch.Tensor:
        dims = len(data), data[0].shape[1], self.window
        tensors = torch.zeros(*dims, dtype=torch.float32, device=self.device)

        for i in range(len(data)):
            block = data[i]
            used_size = self.window if len(block) > self.window else len(block)
            tensors[i, :, :used_size] = torch.from_numpy(block[:used_size, :].T)
        return tensors

    def __del__(self):
        del self.tensor

    def __len__(self) -> int:
        return len(self.tensor)

    def __getitem__(self, idx) -> List:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return [self.tensor[idx]]


class CroppedDataset2D(Dataset):
    def __init__(self, data: np.ndarray, to: str = 'cpu', window: int = 25):
        self.device = to
        self.window = window

        self.tensor = self._prepare_data(data)

    def _prepare_data(self, data: np.ndarray) -> torch.Tensor:
        dims = len(data), 1, data[0].shape[1], self.window
        tensors = torch.zeros(*dims, dtype=torch.float32, device=self.device)

        for i in range(len(data)):
            block = data[i]
            used_size = self.window if len(block) > self.window else len(block)
            tensors[i, 0, :, :used_size] = torch.from_numpy(block[:used_size, :].T)
        return tensors

    def __del__(self):
        del self.tensor

    def __len__(self) -> int:
        return len(self.tensor)

    def __getitem__(self, idx) -> List:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return [self.tensor[idx]]


class CustomMinMaxScaler(MinMaxScaler):
    def __init__(self):
        super().__init__()
        self.x_min = None
        self.x_max = None

    def fit(self, X: List, y=None) -> CustomMinMaxScaler:
        self.x_min = np.min([x.min(axis=0) for x in X], axis=0)
        self.x_max = np.max([x.max(axis=0) for x in X], axis=0)
        return self

    def fit_transform(self, X: List, y: np.array = None, **fit_params) -> np.array:
        return self.fit(X).transform(X)

    def transform(self, X: List) -> np.array:
        # (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        diff = self.x_max - self.x_min
        return np.asarray([(x - self.x_min) / diff for x in X], dtype='object')


class CustomStandardScaler(StandardScaler):
    def __init__(self):
        super().__init__()
        self.std = None
        self.mean = None

    @staticmethod
    def _flatten_dataset(dataset: List) -> np.array:
        ret = np.array([embedding for block in dataset for embedding in block])
        return ret

    def fit(self, X: List, y=None) -> CustomStandardScaler:
        data = self._flatten_dataset(X)
        self.std = data.std(axis=0)
        self.mean = data.mean(axis=0)
        return self

    def fit_transform(self, X: List, y: np.array = None, **fit_params) -> np.array:
        return self.fit(X).transform(X)

    def transform(self, X: List, copy=None) -> np.array:
        # (X - X.mean(axis=0)) / X.std(axis=0)
        return np.asarray([(x - self.mean) / self.std for x in X], dtype='object')

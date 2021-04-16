#
# Original work at: https://github.com/LogAnalysisTeam/methods4logfiles/blob/main/src/models/autoencoder_tcnn.py
#
#
#
#

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sklearn
from tqdm import tqdm
from typing import List, Callable
import sys

from .utils import time_decorator
from .tcnn import TemporalConvNet
from .datasets import CroppedDataset1D, EmbeddingDataset

SEED = 160121

np.random.seed(SEED)
torch.manual_seed(SEED)


class AETCNPyTorch(nn.Module):
    def __init__(self, input_dim: int, window: int, layers: List, kernel_size: int, dropout: float):
        super().__init__()
        assert kernel_size % 2 == 1 and kernel_size > 1
        self.encoder = TemporalConvNet(input_dim, layers[0], kernel_size, dropout, include_last_relu=True)

        self.relu = nn.ReLU()
        flatten_dim = layers[0][-1] * window
        self.fc1 = nn.Linear(flatten_dim, layers[1])
        self.fc2 = nn.Linear(layers[1], flatten_dim)

        self.decoder = TemporalConvNet(layers[0][-1], layers[2], kernel_size, dropout, include_last_relu=False)

    def _bottleneck(self, x: torch.Tensor):
        org_shape = x.size()
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.reshape(x, shape=org_shape)
        return x

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self._bottleneck(x)
        x = self.decoder(x)
        return x


class AETCN(sklearn.base.OutlierMixin):
    def __init__(self, epochs: int = 1, batch_size: int = 32, optimizer: str = 'adam',
                 loss: str = 'mean_squared_error', learning_rate: float = 0.001, dataset_type: str = 'cropped',
                 window: int = 15, verbose: int = True):
        # add dictionary with architecture of the model i.e., number of layers, hidden units per layer etc.
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.dataset_type = dataset_type
        self.window = window
        self.verbose = verbose

        # internal representation of a torch model
        self._model = None
        self._selected_features = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, X: np.ndarray) -> AETCN:
        # 1. convert to torch Tensor
        train_dl = self._numpy_to_tensors(X, batch_size=self.batch_size, shuffle=True)

        # 2. initialize model
        if not self._model:
            self._initialize_model(X[0].shape[-1], [50, 128, X[0].shape[-1]], 3, 0.2)

        loss_function = self._get_loss_function()
        opt = self._get_optimizer()

        for epoch in range(self.epochs):
            self._model.train()
            loss, execution_time = self._train_epoch(train_dl, opt, loss_function)

            if self.verbose:
                digits = int(np.log10(self.epochs)) + 1
                print(f'Epoch: {epoch + 1:{digits}}/{self.epochs}, loss: {loss:.5f}, time: {execution_time:.5f} s')
        return self

    def predict(self, X: np.ndarray) -> np.array:
        test_dl = self._numpy_to_tensors(X, batch_size=128, shuffle=False)

        loss_function = self._get_loss_function(reduction='none')

        self._model.eval()
        with torch.no_grad():
            ret = []
            for (batch,) in test_dl:
                batch = batch.to(self._device)
                ret.extend(torch.mean(loss_function(self._model(batch), batch), (1, 2)).tolist())
            return np.asarray(ret)

    def forward_hook(self) -> Callable:
        def hook(module, model_input, output):
            self._selected_features = output

        return hook

    def extract_features(self, X: np.ndarray) -> np.array:
        test_dl = self._numpy_to_tensors(X, batch_size=128, shuffle=False)

        self._model.fc1.register_forward_hook(self.forward_hook())

        self._model.eval()
        with torch.no_grad():
            ret = []
            for (batch,) in test_dl:
                batch = batch.to(self._device)
                _ = self._model(batch)
                ret.extend(self._selected_features.tolist())
            return np.asarray(ret)

    def set_params(self, **kwargs):
        self.window = kwargs['window']
        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['batch_size']
        self.learning_rate = kwargs['learning_rate']
        self._initialize_model(kwargs['input_shape'], kwargs['layers'], kwargs['kernel_size'], kwargs['dropout'])

    def _initialize_model(self, input_shape: int, layers_out: List, kernel_size: int, dropout: float):
        self._model = AETCNPyTorch(input_shape, self.window, layers_out, kernel_size, dropout)
        self._model.to(self._device)

    def _get_loss_function(self, reduction: str = 'mean') -> nn.Module:
        if self.loss == 'mean_squared_error':
            return nn.MSELoss(reduction=reduction)
        elif self.loss == 'kullback_leibler_divergence':
            return nn.KLDivLoss(reduction=reduction)
        else:
            raise NotImplementedError(f'"{self.loss}" is not implemented.')

    def _get_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer == 'adam':
            return torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError(f'"{self.optimizer}" is not implemented.')

    @staticmethod
    def custom_collate(data: List):
        # randomly shuffle data within a batch
        tensor = data[0]
        indexes = torch.randperm(tensor.shape[0])
        return [tensor[indexes]]  # must stay persistent with PyTorch API

    def _numpy_to_tensors(self, X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        if self.dataset_type == 'variable_sized':
            train_ds = EmbeddingDataset(X, batch_size=batch_size)
            collate_fn = self.custom_collate if shuffle else None
            train_dl = DataLoader(train_ds, batch_size=1, shuffle=shuffle, collate_fn=collate_fn)
        elif self.dataset_type == 'cropped':
            train_ds = CroppedDataset1D(X, window=self.window)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
        else:
            raise NotImplementedError('This dataset preprocessing is not implemented yet.')
        return train_dl

    @time_decorator
    def _train_epoch(self, train_dl: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        loss = 0
        n_seen_examples = 0
        train_dl = tqdm(train_dl, file=sys.stdout, ascii=True, unit='batch')
        for (batch,) in train_dl:
            batch = batch.to(self._device)

            optimizer.zero_grad()

            pred = self._model(batch)
            batch_loss = criterion(pred, batch)

            batch_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1)
            torch.nn.utils.clip_grad_value_(self._model.parameters(), 0.5)
            optimizer.step()

            loss += batch_loss.item() * batch.size(0)
            n_seen_examples += batch.size(0)

            train_dl.set_postfix({'loss': loss / n_seen_examples, 'curr_loss': batch_loss.item()})
        return loss / n_seen_examples

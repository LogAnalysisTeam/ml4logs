from argparse import ArgumentParser
from pathlib import Path

import fasttext
import numpy as np
import torch
import torch.nn.functional as F
import logging

from prediction import LogAnomalyDetection
from model_environment import ModelEnvironment
from utils import Statistics

logger = logging.getLogger(__name__)

class LogClassification:
    # TODO: better switch to "click" argument parsing

    @staticmethod
    def add_arguments(argparser: ArgumentParser):
        argparser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
        argparser.add_argument("--lr_gamma", type=float, default=0.9817, help='learning rate gama')

        argparser.add_argument("--fasttext", type=str, required=True, help='path to fasText model')
        argparser.add_argument("--lstm_layers", type=int, default=1, help='number of LSTM layers')
        argparser.add_argument("--linear_width", type=int, default=300, help='width of hidden dense layers')
        argparser.add_argument("--linear_layers", type=int, default=3, help='number of dense layers')
        argparser.add_argument("--weight", type=float, default=1, help='additional training weight for anomaly sample, to fight unbalanced dataset')
        argparser.add_argument("--layer_norm", type=bool, default=False, help='add layer normalization')
        argparser.add_argument("--grad_clip", type=float, default=None, help='value to which clip gradient')



    ENV_FILE_NAME = "env.bin"
    MODEL_FILE_NAME = "model.pt"
    OPTIM_FILE_NAME = "optimizer_state.pt"

    def __init__(self, args=None, path=None):
        if (args == None) == (path == None):
            raise ValueError('One argument required (args or path)')

        if args:
            self.env = ModelEnvironment(args)
        if path:
            self.env = torch.load(path / LogClassification.ENV_FILE_NAME)
            args = self.env.args
        self.args = args

        self.device = args.device
        self.transform = TextToTensor(args.fasttext)
        self.model = ClassifyModelLstm(self.transform.embedding_dim, args).to(device=self.device)
        weights = torch.Tensor([args.weight]).to(device=args.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, args.lr_gamma)

        if path:
            self._load_model(path)

    def train(self, dataloader):
        """Train model."""
        self.model.train()
        train_loss = 0.0
        for inputs, outputs, label in dataloader:
            # Forward pass and compute loss
            inputs = inputs.to(device=self.device)
            outputs = outputs.to(device=self.device)
            results = self.model(inputs)
            loss = self.criterion(results, outputs)
            train_loss += loss.item()

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.grad_clip:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
        self.scheduler.step()
        return train_loss / len(dataloader)

    def predict(self, inputs):
        """Predict outputs."""
        self.model.eval()
        if self.device.type != 'cpu':
            inputs = inputs.to(device=self.device)

        with torch.no_grad():
            results = self.model(inputs)

        if self.device.type != 'cpu':
            del inputs

        return results

    def evaluate(self, dataloader, threshold, label_by_block=False):
        """Evaluate outputs."""
        stats = Statistics()
        validation_loss = 0

        for inputs, outputs, labels in dataloader:
            inputs = inputs.to(device=self.device)
            outputs = outputs.to(device=self.device)
            labels = labels.to(device='cpu')
            results = self.predict(inputs)
            loss = self.criterion(results, outputs)
            results = results.to(device='cpu')
            validation_loss += loss.to(device='cpu').item()
            for i in range(results.size()[0]):
                LogAnomalyDetection.evaluate_threshold(threshold, stats, results[i], labels[i], label_by_block)

        validation_loss = validation_loss / len(dataloader)
        data = {'stats': stats, 'validation_loss': validation_loss}
        return "Validation loss: %.3f, %s" % (validation_loss, stats.as_string()), data

    def test(self, dataloader, threshold=0.5, label_by_block=False):
        stats = Statistics()

        detection_data = {
            'stats': stats,
            'label_by_block': label_by_block,
            'block_split': [],
            'labels': [],
            'threshold': threshold,
            'thresholds': [(t, Statistics()) for t in np.linspace(0, 1.5 * threshold, num=50)],
            'e': []
        }

        detection_data['thresholds'].append((threshold, stats))

        last_block_index = 0

        logger.info(f"    Computing classification ({'labeled by block' if label_by_block else 'labeled by sample'})")
        for inputs, outputs, labels in dataloader:
            inputs = inputs.to(device=self.device)
            labels = labels.to(device='cpu')
            results = self.predict(inputs).to(device='cpu')

            for i in range(results.size()[0]):
                loss = results[i].numpy()
                label = labels[i].numpy()

                for t, s in detection_data['thresholds']:
                    LogAnomalyDetection.evaluate_threshold(t, s, loss, label, label_by_block)

                last_block_index += len(loss)
                detection_data['block_split'].append(last_block_index)
                detection_data['labels'].extend(label)
                detection_data['e'].extend(loss)

        detection_data['block_split'].pop()
        return detection_data

    def _load_model(self, path: Path):
        state = torch.load(
            path / LogClassification.MODEL_FILE_NAME,
            map_location=self.device
        )
        self.transform.load_state_dict(state.pop('transform'))
        self.model.load_state_dict(state)
        file = path / LogClassification.OPTIM_FILE_NAME
        if file.exists() and file.is_file():
            self.optimizer.load_state_dict(torch.load(file, map_location=self.device))

    def save(self, path: Path, model=True, env=True):
        path.resolve().mkdir(parents=True, exist_ok=True)
        if model:
            state = self.model.state_dict()
            state['transform'] = self.transform.state_dict()
            torch.save(state, path / LogClassification.MODEL_FILE_NAME)
            torch.save(self.optimizer.state_dict(), path / LogClassification.OPTIM_FILE_NAME)
        if env:
            torch.save(self.env, path / LogClassification.ENV_FILE_NAME)


class TextToTensor(object):
    """
    Transform text logs to vector representation using given fasttext model and device.
    """

    def __init__(self, fasttext_model_file, label_by_block=False, timedelta_mean=0, timedelta_std=1):
        self.embedding = fasttext.load_model(fasttext_model_file)
        self.embedding_dim = len(self.embedding['a']) + 1
        self.label_by_block = label_by_block
        self.timedelta_mean = timedelta_mean
        self.timedelta_std = timedelta_std

    def __call__(self, sample):
        logs, label = sample['logs'], sample['label']
        timestamps = np.array(sample.get('timestamps', [0 for _ in range(len(logs))]))

        embeddings = np.zeros((len(logs), self.embedding_dim), dtype=np.float32)
        embeddings[:, 1:] = list(map(self.embedding.get_sentence_vector, logs))

        timestamps = timestamps[1:] - timestamps[:-1]
        timestamps = np.insert(timestamps, 0, 0)
        timestamps[timestamps == 0] = 1  # to make log(0) = 0
        timestamps = (np.log10(timestamps) - self.timedelta_mean) / self.timedelta_std
        embeddings[:, 0] = timestamps

        logs = list(map(torch.from_numpy, embeddings))

        if not self.label_by_block and 'labels' in sample:
            labels = torch.tensor(sample['labels'], dtype=torch.float32)
        else:
            labels = torch.ones(len(logs)) if label else torch.zeros(len(logs))

        return (torch.stack(logs),
                labels.unsqueeze(-1),
                labels)

    def compute_normalization(self, data):
        logger.info("Compution data normalization")
        timedeltas = []
        for sample in data:
            time = np.array(sample['timestamps'])
            timedeltas.extend(time[1:] - time[:-1])
        timedeltas = np.array(timedeltas)
        timedeltas[timedeltas == 0] = 1  # to make log(0) = 0
        timedeltas = np.log10(timedeltas)
        self.timedelta_std = np.std(timedeltas)
        self.timedelta_mean = np.mean(timedeltas)
        logger.info(f"Timedelta mean={self.timedelta_mean} std={self.timedelta_std}")

    def state_dict(self):
        return {
            'timedelta_mean': self.timedelta_mean,
            'timedelta_std': self.timedelta_std
        }

    def load_state_dict(self, state_dict):
        self.timedelta_mean = state_dict['timedelta_mean']
        self.timedelta_std = state_dict['timedelta_std']


class ClassifyModelLstm(torch.nn.Module):

    def __init__(self, embedding_dim, args):
        super().__init__()
        self.lstm = torch.nn.LSTM(embedding_dim, embedding_dim, batch_first=True, num_layers=args.lstm_layers)
        self.linears = torch.nn.ModuleList()

        self.linears.append(torch.nn.Linear(embedding_dim, args.linear_width))
        if args.layer_norm:
            self.linears.append(torch.nn.LayerNorm(args.linear_width))

        for _ in range(1, args.linear_layers):
            self.linears.append(torch.nn.Linear(args.linear_width, args.linear_width))
            if args.layer_norm:
                self.linears.append(torch.nn.LayerNorm(args.linear_width))

        self.linearOut = torch.nn.Linear(args.linear_width, 1)

    def forward(self, logs):
        '''logs is tensor (batch, seq, feature)'''
        x = self.lstm(logs)[0]
        for layer in self.linears:
            x = F.relu(layer(x))
        x = self.linearOut(x)
        if not self.training:
            x = torch.sigmoid(x)
        return x

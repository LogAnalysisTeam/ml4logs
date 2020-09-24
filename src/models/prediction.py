from argparse import ArgumentParser
from pathlib import Path

import fasttext
import numpy as np
import torch
import torch.nn.functional as F
import logging

from model_environment import ModelEnvironment
from utils import Statistics

logger = logging.getLogger(__name__)

def debug_param(p):
    res = {
        'w_mean': torch.mean(p.data).item(),
        'w_med': torch.median(p.data).item(),
        'w_max': torch.max(torch.abs(p.data)).item()
    }
    if p.grad is not None:
        res.update({
            'g_mean': torch.mean(p.grad).item(),
            'g_med': torch.median(p.grad).item(),
            'g_max': torch.max(torch.abs(p.grad)).item()
        })
    return res


class LogAnomalyDetection:

    @staticmethod
    def add_arguments(argparser: ArgumentParser):
        argparser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
        argparser.add_argument("--lr_gamma", type=float, default=0.9817, help='learning rate gama')
        argparser.add_argument("--loss", choices=['cos', 'mse', 'L1'], default='cos',
                               help='loss function used to measure embedding distance')

        argparser.add_argument("--fasttext", type=str, default=None, help='path to fasText model')
        argparser.add_argument("--lstm_layers", type=int, default=1, help='number of LSTM layers')
        argparser.add_argument("--linear_width", type=int, default=300, help='width of hidden dense layers')
        argparser.add_argument("--linear_layers", type=int, default=3, help='number of dense layers')
        argparser.add_argument("--layer_norm", type=bool, default=False, help='add layer normalization')
        argparser.add_argument("--grad_clip", type=float, default=None, help='value to which clip gradient')

    def apply_arguments(self, args):
        setattr(self.args, 'lr', getattr(args, 'lr', 1e-3))
        setattr(self.args, 'lr_gamma', getattr(args, 'lr_gamma', 0.9817))
        setattr(self.args, 'loss', getattr(args, 'loss', 'cos'))
        setattr(self.args, 'lstm_layers', getattr(args, 'lstm_layers', 1))
        setattr(self.args, 'linear_width', getattr(args, 'linear_width', 300))
        setattr(self.args, 'linear_layers', getattr(args, 'linear_layers', 3))
        setattr(self.args, 'layer_norm', getattr(args, 'layer_norm', False))
        setattr(self.args, 'grad_clip', getattr(args, 'grad_clip', None))

        if not self.args.fasttext:
            setattr(self.args, 'fasttext', getattr(args, 'fasttext', None))

        if self.args.fasttext != args.fasttext:
            logger.info(f"WARNING, different fastext path: {self.args.fasttext} -> {args.fasttext}")

    ENV_FILE_NAME = "env.bin"
    MODEL_FILE_NAME = "model.pt"
    OPTIM_FILE_NAME = "optimizer_state.pt"

    def __init__(self, args, path: Path = None, epoch=None):
        self.args = args
        if path:
            self.env = torch.load(path / LogAnomalyDetection.ENV_FILE_NAME)
            self.apply_arguments(self.env.args)
        else:
            self.env = ModelEnvironment(args)

        if not self.args.fasttext:
            logger.info("ERROR missing path to fastext model")
            exit(1)

        self.device = args.device
        self.transform = TextToTensorSeq(args.fasttext)
        self.model = PredictModelLstm(self.transform.embedding_dim, args).to(device=self.device)
        self.criterion = self._create_criterion()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, args.lr_gamma)
        self.can_train = True
        self.epoch = 0
        if path:
            self._load_model(epoch)

    def _create_criterion(self):
        if self.args.loss == 'cos':
            cos = torch.nn.CosineEmbeddingLoss(reduction='none')

            def criterion(x1, x2, l):
                return cos(x1.permute(0, 2, 1), x2.permute(0, 2, 1), ((l * 2) - 1) * -1)

        elif self.args.loss == 'L1':
            l1 = torch.nn.L1Loss(reduction='none')

            def criterion(x1, x2, l):
                return torch.mean(l1(x1, x2), dim=2)

        else:
            mse = torch.nn.MSELoss(reduction='none')

            def criterion(x1, x2, l):
                return torch.mean(mse(x1, x2), dim=2)

        return criterion

    def train(self, dataloader):
        """Train model."""
        assert self.can_train

        self.model.train()
        train_loss = 0.0
        features = self.transform.embedding_dim
        for inputs, outputs, labels in dataloader:
            # Forward pass and compute loss
            inputs = inputs.to(device=self.device)
            outputs = outputs.to(device=self.device)
            labels = labels.to(device=self.device)
            results = self.model(inputs)
            loss = self.criterion(results, outputs, labels)  # labels only used with cos loos + labeled anomal training data experiment
            loss = torch.mean(loss)
            train_loss += loss.item()

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.grad_clip:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
        self.scheduler.step()

        errors = []
        for inputs, outputs, labels in dataloader:
            # Forward pass and compute loss
            inputs = inputs.to(device=self.device)
            outputs = outputs.to(device=self.device)
            labels = labels.to(device=self.device)
            results = self.model(inputs)
            loss = self.criterion(results, outputs, labels)  # labels only used with cos loos + labeled anomal training data experiment
            errors.extend(loss.view(-1).to(device='cpu').tolist())
        self.threshold = np.mean(errors) + 2 * np.std(errors)

        self.epoch += 1

        return train_loss / len(dataloader)

    def evaluate(self, dataloader):
        """Evaluate outputs."""
        self.model.eval()
        total_loss = 0
        features = self.transform.embedding_dim
        with torch.no_grad():
            for inputs, outputs, labels in dataloader:
                inputs = inputs.to(device=self.device)
                outputs = outputs.to(device=self.device)
                labels = labels.to(device=self.device)
                results = self.model(inputs)
                loss = self.criterion(results, outputs, labels)  # labels only used with cos loos + labeled anomal training data experiment
                total_loss += torch.mean(loss).item()

        return total_loss / len(dataloader)

    def predict(self, inputs):
        """Predict outputs."""
        self.model.eval()
        inputs = inputs.to(device=self.device)

        with torch.no_grad():
            results = self.model(inputs)

        return results

    def test_detection(self, dataloader, label_by_block=False, dataset_name=None):

        stats = Statistics()

        detection_data = {
            'stats': stats,
            'label_by_block': label_by_block,
            'block_split': [],
            'labels': [],
            'threshold': self.threshold,
            'thresholds': [(t, Statistics()) for t in np.linspace(0, 1.5 * self.threshold, num=50)],
            'e': []
        }

        detection_data['thresholds'].append((self.threshold, stats))

        last_block_index = 0

        logger.info(f"    Computing and comparing predictions ({'labeled by block' if label_by_block else 'labeled by sample'})")
        for inputs, outputs, labels in dataloader:

            inputs = inputs.to(device=self.device)
            outputs = outputs.to(device='cpu')
            labels = labels.to(device='cpu')
            results = self.predict(inputs).to(device='cpu')
            losses = self.criterion(results, outputs, labels)  # labels only used with cos loos + labeled anomal training data experiment

            for i in range(losses.size()[0]):
                loss = losses[i].numpy()
                label = labels[i].numpy()

                for t, s in detection_data['thresholds']:
                    self.evaluate_threshold(t, s, loss, label, label_by_block)

                last_block_index += len(loss)
                detection_data['block_split'].append(last_block_index)
                detection_data['labels'].extend(label)
                detection_data['e'].extend(loss)

        detection_data['block_split'].pop()
        self.env.add_test_result(self.epoch, dataset_name, detection_data)
        return detection_data

    @staticmethod
    def evaluate_threshold(threshold, stats, errors, label, label_by_block):
        anomal_idxs = np.argwhere(errors > threshold).flatten()
        if label_by_block:
            if label[-1] == 1:  # take last label, beggining can be padded by zeros
                if len(anomal_idxs) > 0:
                    stats.add_tp()
                else:
                    stats.add_fn()
            else:
                if len(anomal_idxs) > 0:
                    stats.add_fp()
                else:
                    stats.add_tn()
        else:
            for idx in range(len(label)):
                if label[idx]:
                    if idx in anomal_idxs:
                        stats.add_tp()
                    else:
                        stats.add_fn()
                else:
                    if idx in anomal_idxs:
                        stats.add_fp()
                    else:
                        stats.add_tn()

    def _load_model(self, epoch=None):
        if epoch == 'best':
            epoch = self.env.best_epoch
        if epoch == None:
            epoch = len(self.env.epochs) - 1
        self.epoch = epoch
        self.can_train = epoch == len(self.env.epochs) - 1
        path = self.args.path / self.env.epochs[epoch]['epoch_dir']

        state = torch.load(
            path / LogAnomalyDetection.MODEL_FILE_NAME,
            map_location=self.device
        )
        self.transform.load_state_dict(state.pop('transform'))
        self.threshold = state.pop('threshold')
        self.model.load_state_dict(state)
        file = path / LogAnomalyDetection.OPTIM_FILE_NAME
        if file.exists() and file.is_file():
            self.optimizer.load_state_dict(torch.load(file, map_location=self.device))

    def save(self, path: Path, model=True, env=True):
        path.resolve().mkdir(parents=True, exist_ok=True)
        if model:
            state = self.model.state_dict()
            state['transform'] = self.transform.state_dict()
            state['threshold'] = self.threshold
            torch.save(state, path / LogAnomalyDetection.MODEL_FILE_NAME)
            torch.save(self.optimizer.state_dict(), path / LogAnomalyDetection.OPTIM_FILE_NAME)
        if env:
            torch.save(self.env, path / LogAnomalyDetection.ENV_FILE_NAME)


class TextToTensorSeq(object):
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
            labels = torch.tensor(sample['labels'][1:])
        else:
            labels = torch.ones(len(logs) - 1) if label else torch.zeros(len(logs) - 1)

        return (torch.stack(logs[:-1]),
                torch.stack(logs[1:]),
                labels)

    def compute_normalization(self, data):
        logger.info("Computing data normalization")
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


class PredictModelLstm(torch.nn.Module):

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

        self.linears.append(torch.nn.Linear(args.linear_width, embedding_dim))

    def forward(self, logs):
        '''logs is tensor (batch, seq, feature)'''
        x = self.lstm(logs)[0]
        for layer in self.linears:
            x = F.relu(layer(x))
        return x

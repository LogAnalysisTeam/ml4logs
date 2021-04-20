# ===== IMPORTS =====
# === Standard library ===
import logging
import pathlib
import json

# === Thirdparty ===
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data as tdata
import torch.nn.functional as tfunctional
import torch.nn.utils.rnn as tutilsrnn
from sklearn.metrics import precision_recall_fscore_support

# === Local ===
import ml4logs


# ===== GLOBALS =====
logger = logging.getLogger(__name__)
NORMAL_LABEL = 0
ABNORMAL_LABEL = 1


# ===== CLASSES =====
class SequenceDataset(tdata.Dataset):
    def __init__(self, inputs, outputs, labels):
        self._inputs = inputs
        self._outputs = outputs
        self._labels = labels

    def __getitem__(self, idx):
        return self._inputs[idx], self._outputs[idx], self._labels[idx]

    def __len__(self):
        return len(self._inputs)


class Seq2SeqModel(torch.nn.Module):
    def __init__(self, f_dim, n_lstm_layers=1,
                 n_hidden_linears=2, linear_width=300, linear_norm=False):
        super().__init__()
        self._lstm = torch.nn.LSTM(f_dim, f_dim, batch_first=True,
                                   num_layers=n_lstm_layers)
        linears = [torch.nn.Linear(f_dim, linear_width)]
        if linear_norm:
            linears.append(torch.nn.LayerNorm(linear_width))
        linears.append(torch.nn.LeakyReLU())
        for _ in range(n_hidden_linears):
            linears.append(torch.nn.Linear(linear_width, linear_width))
            if linear_norm:
                linears.append(torch.nn.LayerNorm(linear_width))
            linears.append(torch.nn.LeakyReLU())
        linears.append(torch.nn.Linear(linear_width, f_dim))
        self._linears = torch.nn.Sequential(*linears)

    def forward(self, X):
        out, _ = self._lstm(X)
        out, lengths = tutilsrnn.pad_packed_sequence(X, batch_first=True)
        return self._linears(out)


class Seq2SeqModelTrainer:
    def __init__(self, device, f_dim, model_kwargs,
                 optim_kwargs, lr_scheduler_kwargs):
        self._model = Seq2SeqModel(f_dim, **model_kwargs).to(device)
        self._criterion = torch.nn.MSELoss()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), **optim_kwargs)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._optimizer, **lr_scheduler_kwargs)
        self._device = device
        self._threshold = 0.0

    def train(self, dataloader):
        self._model.train()
        train_loss = 0.0
        for inputs, outputs, _ in dataloader:
            results, outputs = self._forward(inputs, outputs)
            loss = self._criterion(results, outputs)
            train_loss += loss.item()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
        self._scheduler.step()
        return train_loss / len(dataloader)

    def evaluate(self, dataloader):
        self._model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, outputs, _ in dataloader:
                results, outputs = self._forward(inputs, outputs)
                loss = self._criterion(results, outputs)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def compute_threshold(self, dataloader):
        self._model.eval()
        errors = []
        for inputs, outputs, _ in dataloader:
            results, outputs = self._forward(inputs, outputs)
            loss = tfunctional.mse_loss(results, outputs, reduction='none')
            loss = torch.mean(loss, dim=2)
            loss = torch.mean(loss, dim=1)
            errors.extend(loss.tolist())
        self._threshold = np.mean(errors) + 2 * np.std(errors)
        return self._threshold

    def test(self, dataloader, threshold=None):
        if threshold is None:
            threshold = self._threshold
        self._model.eval()
        labels = []
        result_labels = []
        with torch.no_grad():
            for inputs, outputs, labels_ in dataloader:
                results, outputs = self._forward(inputs, outputs)
                loss = tfunctional.mse_loss(results, outputs, reduction='none')
                loss = torch.mean(loss, dim=2)
                loss = torch.mean(loss, dim=1)
                result_labels_ = torch.where(
                    loss > threshold, ABNORMAL_LABEL, NORMAL_LABEL)
                labels.append(labels_.numpy())
                result_labels.append(result_labels_.to(device='cpu').numpy())
        labels = np.concatenate(labels)
        results = np.concatenate(result_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, results, average='binary', zero_division=0
        )
        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def threshold(self):
        return self._threshold

    def _forward(self, inputs, outputs):
        inputs = inputs.to(self._device)
        outputs = outputs.to(self._device)
        outputs, lengths = tutilsrnn.pad_packed_sequence(
            outputs,
            batch_first=True
        )

        results = self._model(inputs)
        results = tutilsrnn.pack_padded_sequence(
            results,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        results, _ = tutilsrnn.pad_packed_sequence(
            results,
            batch_first=True
        )

        return results, outputs


# ===== FUNCTIONS =====
def train_test_seq2seq(args):
    np.random.seed(args['seed'])
    if args['device'] == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU!")
        args['device'] = 'cpu'

    torch.manual_seed(args['seed'])

    features_path = pathlib.Path(args['features_path'])
    blocks_path = pathlib.Path(args['blocks_path'])
    labels_path = pathlib.Path(args['labels_path'])
    stats_path = pathlib.Path(args['stats_path'])

    ml4logs.utils.mkdirs(files=[stats_path])

    features = np.load(features_path).astype(np.float32)
    blocks = np.load(blocks_path)
    labels = np.load(labels_path)
    logger.info('Loaded features %s', features.shape)
    logger.info('Loaded blocks %s', blocks.shape)
    logger.info('Loaded labels %s', labels.shape)

    blocks_unique, blocks_counts = np.unique(blocks, return_counts=True)
    blocks_used = blocks_unique[blocks_counts > 1]
    logger.info('Blocks used %s', blocks_used.shape)
    normal_blocks = blocks_used[labels[blocks_used] == NORMAL_LABEL]
    test_abnormal_blocks = blocks_used[labels[blocks_used] == ABNORMAL_LABEL]
    logger.info('Normal blocks %s, abnormal blocks %s',
                normal_blocks.shape, test_abnormal_blocks.shape)
    logger.info('Split with train size = %.2f', args['train_size'])
    train_blocks, test_normal_blocks = train_test_split(
        normal_blocks,
        train_size=args['train_size']
    )
    logger.info('Split with validation size = %.2f', args['validation_size'])
    test_blocks = np.concatenate((test_normal_blocks, test_abnormal_blocks))
    train_blocks, validation_blocks = train_test_split(
        train_blocks,
        test_size=args['validation_size']
    )
    logger.info('Train normal blocks %s', train_blocks.shape)
    logger.info('Validation normal blocks %s', validation_blocks.shape)
    logger.info('Test normal blocks %s', test_normal_blocks.shape)
    logger.info('Test abnormal blocks %s', test_abnormal_blocks.shape)

    scaler = StandardScaler()
    logger.info('Fit StandardScaler with train blocks')
    scaler.fit(features[np.isin(blocks, train_blocks)])
    logger.info('Scale whole dataset')
    features_scaled = scaler.transform(features)

    logger.info('Create sequence datasets')
    values = {}
    for block, array in zip(blocks, features_scaled):
        list_ = values.setdefault(block, list())
        list_.append(array)
    values = {block: np.stack(arrays) for block, arrays in values.items()}
    train_dataset = create_sequence_dataset(values, labels, train_blocks)
    validation_dataset = create_sequence_dataset(
        values, labels, validation_blocks)
    test_dataset = create_sequence_dataset(values, labels, test_blocks)

    logger.info('Create torch dataloaders')
    loaders_kwargs = {
        'batch_size': args['batch_size'],
        'collate_fn': pad_collate,
        'shuffle': True,
        'pin_memory': True
    }
    train_l = tdata.DataLoader(train_dataset, **loaders_kwargs)
    validation_l = tdata.DataLoader(validation_dataset, **loaders_kwargs)
    test_l = tdata.DataLoader(test_dataset, **loaders_kwargs)

    logger.info('Create model, optimizer, lr_scheduler and trainer')

    device = torch.device(args['device'])
    f_dim = features.shape[-1]
    trainer = Seq2SeqModelTrainer(
        device,
        f_dim,
        args['model_kwargs'],
        args['optim_kwargs'],
        args['lr_scheduler_kwargs']
    )

    stats = {
        'step': args,
        'metrics': {'train': [], 'test': []}
    }

    logger.info('Start training')
    validation_loss = trainer.evaluate(validation_l)
    stats['metrics']['train'].append(
        {'epoch': 0, 'validation_loss': validation_loss})
    logger.info('Epoch: %3d | Validation loss: %.2f', 0, validation_loss)
    for epoch in range(1, args['epochs'] + 1):
        train_loss = trainer.train(train_l)
        validation_loss = trainer.evaluate(validation_l)
        trainer.compute_threshold(train_l)
        stats['metrics']['train'].append(
            {'epoch': epoch,
             'train_loss': train_loss,
             'validation_loss': validation_loss}
        )
        logger.info('Epoch: %3d | Train loss: %.2f | Validation loss: %.2f',
                    epoch, train_loss, validation_loss)

    logger.info('Start testing using different thresholds')
    thresholds = np.linspace(0, 1.5 * trainer.threshold(), num=10)
    for threshold in thresholds:
        info = trainer.test(test_l, threshold)
        stats['metrics']['test'].append(info)
        logger.info(' | '.join([
            'Threshold = {threshold:.2f}',
            'Precision = {precision:.2f}',
            'Recall = {recall:.2f}',
            'F1-score = {f1:.2f}',
        ]).format(**info))

    logger.info('Save metrics into \'%s\'', stats_path)
    stats_path.write_text(json.dumps(stats, indent=4))


def create_sequence_dataset(values, labels_, blocks):
    inputs = []
    outputs = []
    labels = []
    for block in blocks:
        inputs.append(values[block][:-1])
        outputs.append(values[block][1:])
        labels.append(labels_[block])
    return SequenceDataset(inputs, outputs, labels)


def pad_collate(samples):
    inputs, outputs, labels = tuple(zip(*samples))
    inputs = tuple(map(torch.from_numpy, inputs))
    outputs = tuple(map(torch.from_numpy, outputs))
    labels = torch.tensor(labels)
    inputs = tutilsrnn.pack_sequence(inputs, enforce_sorted=False)
    outputs = tutilsrnn.pack_sequence(outputs, enforce_sorted=False)
    return inputs, outputs, labels

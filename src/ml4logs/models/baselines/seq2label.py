# ===== IMPORTS =====
# === Standard library ===
from collections import defaultdict, Counter
import logging
import pathlib
from pathlib import Path
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
from ml4logs.features.utils import load_features_as_dict
from ml4logs.data.hdfs import load_labels

# ===== GLOBALS =====
logger = logging.getLogger(__name__)
NORMAL_LABEL = 0
ABNORMAL_LABEL = 1


# ===== CLASSES =====
class Seq2LabelModelTrainer:
    def __init__(self, device, f_dim, model_kwargs,
                 optim_kwargs, lr_scheduler_kwargs):
        self._model = ml4logs.models.baselines.SeqModel(
            f_dim, **model_kwargs
        ).to(device)
        self._criterion = torch.nn.BCEWithLogitsLoss()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), **optim_kwargs)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._optimizer, **lr_scheduler_kwargs)
        self._device = device

    def train(self, dataloader):
        self._model.train()
        train_loss = 0.0
        for inputs, labels in dataloader:
            results, labels = self._forward(inputs, labels)
            loss = self._criterion(results, labels)
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
            for inputs, labels in dataloader:
                results, labels = self._forward(inputs, labels)
                loss = self._criterion(results, labels)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def test(self, dataloader, threshold):
        self._model.eval()
        labels = []
        result_labels = []
        with torch.no_grad():
            for inputs, labels_ in dataloader:
                results, labels_ = self._forward(inputs, labels_)
                results = torch.sigmoid(results)
                result_labels_ = torch.where(
                    results > threshold, ABNORMAL_LABEL, NORMAL_LABEL)
                result_labels_ = torch.where(
                    torch.sum(result_labels_, dim=1) > 0,
                    ABNORMAL_LABEL,
                    NORMAL_LABEL
                )
                labels_ = torch.where(
                    torch.sum(labels_, dim=1) > 0,
                    ABNORMAL_LABEL,
                    NORMAL_LABEL
                )
                labels.append(labels_.to(device='cpu').numpy())
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

    def _forward(self, inputs, labels):
        # gets data as created by `pad_collate()`

        inputs = inputs.to(self._device)
        labels = labels.to(self._device)

        # labels shape will be (batch_size, max_sequence_length)
        # lengths will be (batch_size, ) tensor with actual sequence lengths
        labels, lengths = tutilsrnn.pad_packed_sequence(
            labels,
            batch_first=True        )

        # results will be (batch_size, max_sequence_length, 1) - there is a single output neuron
        outputs = self._model(inputs)

        # the network predicts even after actual sequence_length
        # the following pack_padded and pad_packed will set all these predictions to 0
        # so they do not mess with loss (targets are also padded b zeros)
        outputs = tutilsrnn.pack_padded_sequence(
            outputs,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        outputs, _ = tutilsrnn.pad_packed_sequence(
            outputs,
            batch_first=True
        )

        # squeeze removes the last dimension so we get (batch_size, max_sequence_length)
        return torch.squeeze(outputs), labels


# ===== FUNCTIONS =====
def train_test_seq2label(args):
    np.random.seed(args['seed'])
    if args['device'] == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU!")
        args['device'] = 'cpu'

    torch.manual_seed(args['seed'])

    train_path = Path(args['train_path'])
    val_path = Path(args['val_path'])
    test_path = Path(args['test_path'])
    stats_path = Path(args['stats_path'])

    ml4logs.utils.mkdirs(files=[stats_path])

    def load_split(input_path, label_path):
        logger.info(
            f'Loading split:\n\t"{args[input_path]}"\n\t"{args[label_path]}"')
        labels = load_labels(args[label_path])
        inputs = load_features_as_dict(args[label_path], args[input_path])
        logger.info(
            f" # input blocks: {len(inputs)}, labels shape: {labels.shape}")
        return inputs, labels

    train_blocks, train_labels = load_split("train_path", "train_label_path")
    val_blocks, val_labels = load_split("val_path", "val_label_path")
    test_blocks, test_labels = load_split("test_path", "test_label_path")

    # originally implemented splits:
    #   train - only normal blocks
    #   val - only normal blocks
    #   test - rest of normal blocks and all anomalous

    train_dataset = create_sequence_dataset(train_blocks, train_labels)
    validation_dataset = create_sequence_dataset(val_blocks, val_labels)
    test_dataset = create_sequence_dataset(test_blocks, test_labels)

    logger.info('Creating Torch DataLoaders')
    loaders_kwargs = {
        'batch_size': args['batch_size'],
        'collate_fn': pad_collate,
        'shuffle': True,
        'pin_memory': True
    }
    train_l = tdata.DataLoader(train_dataset, **loaders_kwargs)
    validation_l = tdata.DataLoader(validation_dataset, **loaders_kwargs)
    test_l = tdata.DataLoader(test_dataset, **loaders_kwargs)

    logger.info('Creating model, optimizer, lr_scheduler and trainer')
    device = torch.device(args['device'])
    f_dim = train_blocks[list(train_blocks.keys())[0]].shape[-1]
    trainer = Seq2LabelModelTrainer(
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

    logger.info('Starting training')
    validation_loss = trainer.evaluate(validation_l)
    stats['metrics']['train'].append(
        {'epoch': 0, 'validation_loss': validation_loss})
    logger.info('Epoch: %3d | Validation loss: %.2f', 0, validation_loss)

    for epoch in range(1, args['epochs'] + 1):
        train_loss = trainer.train(train_l)
        validation_loss = trainer.evaluate(validation_l)
        stats['metrics']['train'].append(
            {'epoch': epoch,
             'train_loss': train_loss,
             'validation_loss': validation_loss}
        )
        logger.info('Epoch: %3d | Train loss: %.2f | Validation loss: %.2f',
                    epoch, train_loss, validation_loss)

    logger.info('Start testing using different thresholds')
    thresholds = np.linspace(0, 1.0, num=10)
    for threshold in thresholds:
        info = trainer.test(test_l, threshold)
        stats['metrics']['test'].append(info)
        logger.info(' | '.join([
            'Threshold = {threshold:.2f}',
            'Precision = {precision:.2f}',
            'Recall = {recall:.2f}',
            'F1-score = {f1:.2f}',
        ]).format(**info))

    logger.info('Saving metrics into \'%s\'', stats_path)
    stats_path.write_text(json.dumps(stats, indent=4))


def create_sequence_dataset(blocks, labels_):
    inputs = []
    labels = []
    for block, label in zip(blocks.values(), labels_.Label.values):
        inputs.append(block.astype(np.float32, copy=False))
        labels.append(
            torch.ones(block.shape[0], dtype=torch.float32)
            if label
            else torch.zeros(block.shape[0], dtype=torch.float32)
        )
    return ml4logs.models.baselines.SequenceDataset(inputs, labels)


def pad_collate(samples):
    # samples: list of (input,lable) tuples:
    #   input is (block_size, feature_dim) numpy array 
    #   lable is (block_size,) torch tensor

    # get separate input and label array lists
    inputs, labels = zip(*samples)
    # logger.info(Counter([l.shape[0] for l in labels]))

    # convert input numpy tensors to the torch ones
    inputs = tuple(map(torch.from_numpy, inputs))

    # pack everything
    inputs = tutilsrnn.pack_sequence(inputs, enforce_sorted=False)
    labels = tutilsrnn.pack_sequence(labels, enforce_sorted=False)

    return inputs, labels

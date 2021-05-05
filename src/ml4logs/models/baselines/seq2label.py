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
from ml4logs.models.utils import classify, get_metrics, get_threshold_metrics, find_optimal_threshold

# ===== GLOBALS =====
logger = logging.getLogger(__name__)
NORMAL_LABEL = 0
ABNORMAL_LABEL = 1


# ===== CLASSES =====
class Seq2LabelModelTrainer:
    def __init__(self, device, f_dim, many_to_one, model_kwargs,
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
        self._many_to_one = many_to_one

    def train(self, dataloader):
        self._model.train()
        train_loss = 0.0
        for inputs, labels, lengths in dataloader:
            results, labels, _ = self._forward(inputs, labels, lengths)
            loss = self._criterion(results, labels)
            train_loss += loss.item()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
        self._scheduler.step()
        return train_loss / len(dataloader)

    def predict_flatten(self, dataloader):
        # makes predictions and returns them and targets both flattened as 1D numpy arrays
        self._model.eval()
        Ys = []
        Ts = []
        with torch.no_grad():
            for X, T, L in dataloader:
                Y, T, _ = self._forward(X, T, L)
                Ys.append(Y.data.to(device='cpu').numpy().reshape(-1))
                Ts.append(T.data.to(device='cpu').numpy().reshape(-1))
        Ys = np.concatenate(Ys)
        Ts = np.concatenate(Ts)
        # logger.info(f"{Ys.shape}, {Ts.shape}")
        return Ys, Ts

    def evaluate(self, dataloader):
        self._model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels, lengths in dataloader:
                results, labels, _ = self._forward(inputs, labels, lengths)
                loss = self._criterion(results, labels)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def find_optimal_threshold(self, dataloader):
        Y, T = self.predict_flatten(dataloader)
        return find_optimal_threshold(T, Y)
        
    # def evaluate_validation_metrics(self, dataloader):
    #     threshold = self.find_optimal_threshold(dataloader)
    #     Y = self.score(dataloader)[:, 1]
    #     C = classify(Y, threshold)
    #     T = self.targets(dataloader)[:, 1]
    #     metrics = get_metrics(T, C)
    #     metrics.update(get_threshold_metrics(T, Y))
    #     return metrics

    def _forward(self, X, T, L):
        if self._many_to_one:
            return self._forward_many_to_one(X, T, L)
        else:
            # lengths can be infered from targets T
            return self._forward_many_to_many(X, T)

    def _forward_many_to_one(self, X, T, L):
        # gets data as created by `pad_collate()`
        X = X.to(self._device)
        T = T.to(self._device)

        # self._model output has shape: (batch_size, max_sequence_length, 1) - there is a single output neuron
        Y = self._model(X)
        Y = Y.reshape(Y.shape[0], Y.shape[1]) # reshape to (batch_size, max_sequence_length)
        Y = Y[range(Y.shape[0]), L - 1] # L - 1: indices of last elements of each output sequence

        return Y, T, L

    def _forward_many_to_many(self, X, T):
        # gets data as created by `pad_collate()`
        X = X.to(self._device)
        T = T.to(self._device)

        # T (labels) shape will be (batch_size, max_sequence_length)
        # lengths will be (batch_size, ) tensor with actual sequence lengths
        T, lengths = tutilsrnn.pad_packed_sequence(
            T,
            batch_first=True)

        # results will be (batch_size, max_sequence_length, 1) - there is a single output neuron
        Y = self._model(X)

        # the network predicts even after actual sequence_length
        # the following pack_padded and pad_packed will set all these predictions to 0
        # so they do not mess with loss (targets are also padded b zeros)
        Y = tutilsrnn.pack_padded_sequence(
            Y,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        Y, _ = tutilsrnn.pad_packed_sequence(
            Y,
            batch_first=True
        )

        # squeeze removes the last dimension so we get (batch_size, max_sequence_length)
        return torch.squeeze(Y), T, lengths


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
            f" # input blocks: {len(inputs)}, # labels: {len(labels)}")
        return inputs, labels

    train_blocks, train_labels = load_split("train_path", "train_label_path")
    val_blocks, val_labels = load_split("val_path", "val_label_path")
    test_blocks, test_labels = load_split("test_path", "test_label_path")

    # originally implemented splits:
    #   train - only normal blocks
    #   val - only normal blocks
    #   test - rest of normal blocks and all anomalous

    many_to_one = args.get("many_to_one", True)
    create_sequence = create_sequence_dataset_many_to_one if many_to_one else create_sequence_dataset_many_to_many
    train_dataset = create_sequence(train_blocks, train_labels)
    validation_dataset = create_sequence(val_blocks, val_labels)
    test_dataset = create_sequence(test_blocks, test_labels)

    logger.info('Creating Torch DataLoaders')
    loaders_kwargs = {
        'batch_size': args['batch_size'],
        'collate_fn': pad_collate_many_to_one if args.get("many_to_one", True) else pad_collate_many_to_many,
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
        many_to_one,
        args['model_kwargs'],
        args['optim_kwargs'],
        args['lr_scheduler_kwargs'],
    )

    method_label = "lstm_classifier_m2o" if many_to_one else "lstm_classifier_m2m"
    stats = {
        'step': args,
        'training': {method_label: []},
        'metrics': {}
    }

    logger.info('Starting training')
    validation_loss = trainer.evaluate(validation_l)
    stats['training'][method_label].append(
        {'epoch': 0, 'validation_loss': validation_loss})
    logger.info('Epoch: %3d | Validation loss: %.2f', 0, validation_loss)

    for epoch in range(1, args['epochs'] + 1):
        train_loss = trainer.train(train_l)
        validation_loss = trainer.evaluate(validation_l)
        stats['training'][method_label].append(
            {'epoch': epoch,
             'train_loss': train_loss,
             'validation_loss': validation_loss}
        )
        logger.info('Epoch: %3d | Train loss: %.2f | Validation loss: %.2f',
                    epoch, train_loss, validation_loss)

    logger.info(f'Computing threshold on validation set')
    threshold, f1 = trainer.find_optimal_threshold(validation_l)
    logger.info(f'Threshold = {threshold}, F1 = {f1}')

    y_test, t_test = trainer.predict_flatten(test_l)
    c_test = classify(y_test, threshold)
    logger.info(f"y_test = {y_test.shape}, c_test = {c_test.shape}, t_test = {t_test.shape}")
    metrics = get_metrics(t_test, c_test)
    metrics.update(get_threshold_metrics(t_test, y_test))

    logger.info(f'Precision = {metrics["precision"]:.2f}, Recall = {metrics["recall"]:.2f}, F1-score = {metrics["f1"]:.2f}')
    logger.info(f'MCC = {metrics["mcc"]:.2f}')
    logger.info(f'AUC = {metrics["auc"]:.2f}, AP = {metrics["ap"]:.2f}')

    stats['metrics'][method_label] = metrics

    logger.info('Saving metrics into \'%s\'', stats_path)
    stats_path.write_text(json.dumps(stats, indent=4))

def create_sequence_dataset_many_to_one(blocks, labels_):
    inputs = []
    for block in blocks.values():
        inputs.append(block.astype(np.float32, copy=False))

    labels = torch.FloatTensor(labels_.Label.values)
    return ml4logs.models.baselines.SequenceDataset(inputs, labels)

def create_sequence_dataset_many_to_many(blocks, labels_):
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


def pad_collate_many_to_one(samples):
    # samples: list of (input,lable) tuples:
    #   input is (block_size, feature_dim) numpy array 
    #   lable is a block label (tensor(1) or tensor(0)) 

    inputs, labels = zip(*samples)
    lengths = np.array([len(i) for i in inputs])
    # convert input numpy tensors to the torch ones
    inputs = tuple(map(torch.from_numpy, inputs))
    # pack inputs
    inputs = tutilsrnn.pack_sequence(inputs, enforce_sorted=False)

    return inputs, torch.FloatTensor(labels), lengths

def pad_collate_many_to_many(samples):
    # samples: list of (input,lable) tuples:
    #   input is (block_size, feature_dim) numpy array 
    #   lable is (block_size,) torch tensor

    # get separate input and label array lists
    inputs, labels = zip(*samples)
    lengths = np.array([len(i) for i in inputs])

    # convert input numpy tensors to the torch ones
    inputs = tuple(map(torch.from_numpy, inputs))

    # pack everything
    inputs = tutilsrnn.pack_sequence(inputs, enforce_sorted=False)
    labels = tutilsrnn.pack_sequence(labels, enforce_sorted=False)

    return inputs, labels, lengths

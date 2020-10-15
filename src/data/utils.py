import numpy as np
from os.path import join as pjoin
import logging
import torch

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def load_labels(file):
    labels = {}
    for line in open(file, 'r'):
        if line.startswith('BlockId'): continue
        blk, truth = line.split(',')
        labels[blk] = "Anomal" in truth
    return labels


def padd_batch(list_of_samples):
    inputs = []
    outputs = []
    labels = []
    for i, o, l in list_of_samples:
        inputs.append(i)
        outputs.append(o)
        labels.append(l)

    batch = (
        # torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True),
        pad_sequence(inputs, batch_first=True),
        pad_sequence(outputs, batch_first=True),
        pad_sequence(labels, batch_first=True, block_labels=True)
    )

    return batch


def padd_batch_classify(list_of_samples):
    inputs = []
    outputs = []
    labels = []
    for i, o, l in list_of_samples:
        inputs.append(i)
        outputs.append(o)
        labels.append(l)

    batch = (
        pad_sequence(inputs, batch_first=True),
        torch.stack(outputs),
        labels
    )

    return batch


def pad_sequence(sequences, batch_first=False, padding_value=0, block_labels=False):
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        offset = max_len - tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, offset:, ...] = tensor
            if block_labels:
                out_tensor[i, :offset, ...] = torch.ones(offset) if tensor[0].item() else torch.zeros(offset)
        else:
            out_tensor[offset:, i, ...] = tensor
            if block_labels:
                out_tensor[:offset, i, ...] = torch.ones(offset) if tensor[0].item() else torch.zeros(offset)

    return out_tensor


class LogBlocksDataset(Dataset):

    def __init__(self, blocks, transform=None):
        self.blocks = blocks
        self.transform = transform

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.blocks[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class LogBlocksSequancesDataset(Dataset):

    def __init__(self, blocks, transform=None):
        self.transform = transform
        self.blocks = blocks
        self.sample2block = []
        offset = 0
        for i, blk in enumerate(self.blocks):
            blk['subsequence_offset'] = offset
            subsequence_count = len(blk['logs']) - 1
            offset += subsequence_count
            self.sample2block.extend([i] * subsequence_count)
        self.len = offset
        logger.info(f'    loaded {len(self.blocks)} sequences with {self.len} subsequences')

    def generate_sample_from_idx(self, idx):
        # block = next(blk for blk in reversed(self.blocks) if blk['subsequence_offset'] < idx)
        # i = idx - block['subsequence_offset'] + 1

        # block = self.blocks[0]
        # for blk in self.blocks:
        #    if blk['subsequence_offset'] > idx:
        #        break
        #    block = blk   
        # i = idx - block['subsequence_offset'] + 2

        block = self.blocks[self.sample2block[idx]]
        i = idx - block['subsequence_offset'] + 2

        log_sequence = block['logs']
        log_timestamps = block.get('timestamps', None)

        sample = {
            'blk': block['blk'],
            'label': block['label'],
            'logs': log_sequence[:i],
        }

        if log_timestamps:
            sample['timestamps'] = log_timestamps[:i]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.generate_sample_from_idx(idx)

"""
Helper used by HDFS and BGL loaders.
"""
def save_for_fulltext_detection(block_dict, blk_train, blk_test, save_path, rng, val_ratio=0.8):
    logger.info('saving datasets for full text detection')
    logger.info(f'train-validation split ratio: {val_ratio}')
    import torch
    validation_idx = int(len(blk_train) * val_ratio)
    rng.shuffle(blk_train)
    baf = pjoin(save_path, 'benchmark_anomal.pt')
    bf = pjoin(save_path, 'benchmark.pt')
    torch.save({
        'train': [block_dict[blk] for blk in blk_train[0:validation_idx]],
        'validation': [block_dict[blk] for blk in blk_train[validation_idx:]],
        'test': [block_dict[blk] for blk in blk_test]
    }, baf)
    torch.save({
        'train': [block_dict[blk] for blk in blk_train[0:validation_idx] if not block_dict[blk]['label']],
        'validation': [block_dict[blk] for blk in blk_train[validation_idx:] if not block_dict[blk]['label']],
        'test': [block_dict[blk] for blk in blk_test]
    }, bf)

"""
Dataset splitting for HDFS and BGL. Code from https://github.com/logpai/loglizer
"""
def split_data(x_data, y_data=None, id_data=None, train_ratio=0, split_type='uniform', rng=np.random.RandomState(1234)):
    assert split_type in ["uniform", "sequential"]
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        
        x_pos = x_data[pos_idx] # all positive samples
        y_pos = y_data[pos_idx]
        id_pos = id_data[pos_idx]

        x_neg = x_data[~pos_idx] # all negative
        y_neg = y_data[~pos_idx]
        id_neg = id_data[~pos_idx]

        train_pos = int(train_ratio * x_pos.shape[0]) # select the same ratio for positive and negative samples
        train_neg = int(train_ratio * x_neg.shape[0])

        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        id_train = np.hstack([id_pos[0:train_pos], id_neg[0:train_neg]])

        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
        id_test = np.hstack([id_pos[train_pos:], id_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    # Random shuffle
    indexes = np.arange(x_train.shape[0])
    rng.shuffle(indexes)
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]

    if id_data is not None:
        return (x_train, y_train), (x_test, y_test), (id_train, id_test)

    return (x_train, y_train), (x_test, y_test)
import os
import pandas as pd
from os.path import join as pjoin
import numpy as np
import re
from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm

from .utils import save_for_fulltext_detection, split_data, slice_sessions

import logging
logger = logging.getLogger(__name__)

# Migrated from loglizer fork
# code based on https://github.com/logpai/loglizer
def load_HDFS(log_file, label_file=None, window='session', train_ratio=0.5, split_type='sequential', save_csv=False, window_size=0, save_path='./', seed=1234, first=None):
    """ Load HDFS structured log into train and test data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.
        seed: random seed used to split and shuffle the data
        first: takes only this numberof records of the log_file (for debugging), all if None

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
    """

    logger.info('====== Input data summary ======')
    logger.info('log_file: {}'.format(log_file))
    logger.info('label_file: {}'.format(label_file))
    logger.info('seed: {}'.format(seed))
    rng = np.random.RandomState(seed)

    if log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file)
        x_data = data['x_data']
        y_data = data['y_data']
        # TODO check for error, missing id data?
        # it seems NPZ import is not used anymore for HDFS
        (x_train, y_train), (x_test, y_test) = split_data(x_data=x_data, y_data=y_data, train_ratio=train_ratio, split_type=split_type, rng=rng)

    elif log_file.endswith('.csv'):
        assert window == 'session', "only window=session is supported for HDFS dataset."
        logger.info("Loading {}".format(log_file))
        struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
        if first is not None:
            struct_log = struct_log.head(first) 
        data_dict = OrderedDict()
        block_dict = {}
        for idx, row in tqdm(struct_log.iterrows(), total=len(struct_log)):
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                    block_dict[blk_Id] = {'blk':blk_Id, 'logs':[], 'timestamps':[]}
                data_dict[blk_Id].append(row['EventId'])
                block_dict[blk_Id]['logs'].append(f"{row['Level']} {row['Component']}: {row['Content']}")
                block_dict[blk_Id]['timestamps'].append(datetime.strptime(f"{row['Date']:06d} {row['Time']:06d}", '%d%m%y %H%M%S').timestamp())

        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
        
        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
            for blk_Id in block_dict.keys():
                block_dict[blk_Id]['label'] = label_dict[blk_Id] == 'Anomaly'

            # Split train and test data
            (x_train, y_train), (x_test, y_test), (blk_train, blk_test) = split_data(data_df['EventSequence'].values, 
                data_df['Label'].values, data_df['BlockId'], train_ratio, split_type, rng=rng)

            save_for_fulltext_detection(block_dict, blk_train, blk_test, save_path, rng=rng)

            logger.info("y_train sum: {}, y_test sum: {}".format(y_train.sum(), y_test.sum()))

        if save_csv:
            data_df.to_csv(pjoin(save_path, 'data_instances.csv'), index=False)

        if window_size > 0:
            x_train, window_y_train, y_train = slice_sessions(x_train, y_train, window_size)
            x_test, window_y_test, y_test = slice_sessions(x_test, y_test, window_size)
            log = "{} {} windows ({}/{} anomaly), {}/{} normal"
            logger.info(log.format("Train:", x_train.shape[0], y_train.sum(), y_train.shape[0], (1-y_train).sum(), y_train.shape[0]))
            logger.info(log.format("Test:", x_test.shape[0], y_test.sum(), y_test.shape[0], (1-y_test).sum(), y_test.shape[0]))
            return (x_train, window_y_train, y_train), (x_test, window_y_test, y_test)

        if label_file is None:
            if split_type == 'uniform':
                split_type = 'sequential'
                logger.warning('only split_type=sequential is supported \
                if label_file=None, switching to sequential.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _) = split_data(x_data, train_ratio=train_ratio, split_type=split_type, rng=rng)
            logger.info('Total: {} instances, train: {} instances, test: {} instances'.format(
                  x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_train, None), (x_test, None), data_df
    else:
        raise NotImplementedError('load_HDFS() only support csv and npz files!')

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    logger.info('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    logger.info('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    logger.info('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test)


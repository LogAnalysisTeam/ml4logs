import os
import pandas as pd
from os.path import join as pjoin
import numpy as np
import re
from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm

from .utils import save_for_fulltext_detection, split_data

import logging
logger = logging.getLogger(__name__)

# Migrated from loglizer fork

def load_BGL(log_file, save_path='./', window='sliding', time_interval=1, stepping_size=1, 
             train_ratio=0.8, seed=1234):
    """
        FIX
        time_interval: length of window in hours
        stepping_size: shift (in hours) between successive windows, for stepping_size > time_interval the successive windows do not overlap

    """
    logger.info('====== Input data summary ======')
    logger.info('log_file: {}'.format(log_file))
    logger.info('seed: {}'.format(seed))
    rng = np.random.RandomState(seed)

    prefile = pjoin(save_path, "train_test_preprocessed.npz")
    if os.path.isfile(prefile):
        logger.info("preprocessed file found, loading {}".format(prefile))
        f = np.load(prefile, allow_pickle=True)
        return (f['x_train'], f['y_train']), (f['x_test'], f['y_test'])
    
    # load Drain preprocessed log csv
    logger.info("loading BGL log file: {}".format(log_file))
    struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
    struct_log['Label'] = struct_log['Label'].apply(lambda x: 0 if x == '-' else 1)
    label_time_pairs = np.array(list(zip(struct_log['Label'],struct_log['Timestamp'])))
    
    logger.info('anomalies: {}'.format(sum(struct_log["Label"])))
    # TODO this configures bgl_preprocess_data but it stores it results at the same time, improve 
    params = {
        'save_path': save_path,
        'window_size': time_interval,
        'step_size': stepping_size
    }

    logger.info("save path for preprocessed files: {}".format(save_path))
    logger.info("preprocessing...")
    event_count_matrix, labels = bgl_preprocess_data(params, label_time_pairs, struct_log)

    logger.info("splitting...")
    (x_train, y_train), (x_test, y_test), (id_train, id_test) = split_data(event_count_matrix, labels, np.arange(labels.shape[0]), train_ratio, rng=rng)
    # x_train is ndarray of Python lists of event ids 
    logger.info("x_train: {}".format(x_train.shape))
    logger.info("y_train: {}".format(y_train.shape))
    logger.info("x_test: {}".format(x_test.shape))
    logger.info("y_test: {}".format(y_test.shape))
    logger.info("saving to: {}".format(prefile))
    np.savez_compressed(prefile, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    logger.info('transforming data for full text prediction')
    block_dict = {}
    with open(params['sliding_cache_file']) as f:
        for i, line in tqdm(enumerate(f), total=params["inst_number"]):
            block = {'blk':i, 'logs':[], 'timestamps':[], 'labels':[]}
            s,e = line.split(',')
            anomal = False
            for log_id in range(int(s), int(e)):
                anomal |= struct_log['Label'][log_id] == 1
                block['labels'].append(struct_log['Label'][log_id])
                block['timestamps'].append(struct_log['Timestamp'][log_id])
                block['logs'].append(' '.join(struct_log.loc[log_id][['Id2','User','Group','Level','Content']]))
            block['label'] = anomal
            block_dict[i] = block

    save_for_fulltext_detection(block_dict, id_train, id_test, save_path, rng=rng)

    return (x_train, y_train), (x_test, y_test)



def bgl_preprocess_data(para, raw_data, event_mapping_data):
    """ split logs into sliding windows, built an event sequences and get the corresponding label

    Args:
    --------
    para: the parameters dictionary
    raw_data: numpy array having rows: (label, time)
    event_mapping_data: a list of event index, where each row index indicates a corresponding log

    Returns:
    --------
    event_count_matrix: event count matrix, where each row is an instance (log sequence vector)
    labels: a list of labels, 1 represents anomaly
    """

    # create the directory for saving the sliding windows (start_index, end_index), which can be directly loaded in future run
    if not os.path.exists(para['save_path']):
        os.mkdir(para['save_path'])
    log_size = raw_data.shape[0]
    sliding_file_path = pjoin(para['save_path'], 'sliding_ws{}_ss{}.csv'.format(para['window_size'], para['step_size']))
    para['sliding_cache_file'] = sliding_file_path

    #=============divide into sliding windows=========#
    start_end_index_list = [] # list of tuples, tuple contains two numbers, 
                              # which represent the start and end of sliding time window
    label_data, time_data = raw_data[:,0], raw_data[:, 1]
    if not os.path.exists(sliding_file_path):
        # split into sliding window
        start_time = time_data[0]
        start_index = 0
        end_index = 0

        # get the first window's start and end indices
        # having end time so the window spans at most "window_size" hours 
        for cur_time in time_data:
            if  cur_time < start_time + para['window_size']*3600:
                end_index += 1
                end_time = cur_time
            else:
                start_end_pair=tuple((start_index, end_index))
                start_end_index_list.append(start_end_pair)
                break
        # move the start and end indices by "step_size" hours for the next sliding windows
        while end_index < log_size:
            start_time = start_time + para['step_size']*3600
            end_time = end_time + para['step_size']*3600
            for i in range(start_index,end_index):
                if time_data[i] < start_time:
                    i+=1
                else:
                    break
            for j in range(end_index, log_size):
                if time_data[j] < end_time:
                    j+=1
                else:
                    break
            start_index = i
            end_index = j
            if end_index-start_index > 2:
                start_end_pair = tuple((start_index, end_index))
                start_end_index_list.append(start_end_pair)
        inst_number = len(start_end_index_list)
        logger.info('there are %d instances (sliding windows) in this dataset\n' % inst_number)
        np.savetxt(sliding_file_path,start_end_index_list, delimiter=',', fmt='%d')
    else:
        logger.info('Loading start_end_index_list from file')
        start_end_index_list = pd.read_csv(sliding_file_path, header=None).values
        inst_number = len(start_end_index_list)
        logger.info('there are %d instances (sliding windows) in this dataset' % inst_number)

    para["inst_number"] = inst_number

    # get all the log indexes in each time window by ranging from start_index to end_index
    expanded_indexes_list=[]
    for t in range(inst_number):
        index_list = []
        expanded_indexes_list.append(index_list)
    for i in range(inst_number):
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)

    event_mapping_data = event_mapping_data['EventId']
    event_num = len(list(set(event_mapping_data)))
    logger.info('there are %d log events' % event_num)

    #=============get labels and event count of each sliding window =========#
    labels = []
    sequences = [] #np.empty((inst_number), dtype=object)
    for j in range(inst_number):
        label = 0   #0 represent success, 1 represent failure
        sequence = []
        for k in expanded_indexes_list[j]:
            sequence.append(event_mapping_data[k])
            if label_data[k]:
                label = 1
                continue
        labels.append(label)
        sequences.append(sequence)
    assert inst_number == len(labels)
    logger.info("among %d instances, there are %d anomalies"%(len(sequences), sum(labels)))
    lenghts = list(map(len,sequences))
    logger.info("instances len mean=%d median=%d"%(np.mean(lenghts), np.median(lenghts)))
    return np.array(sequences), np.array(labels)
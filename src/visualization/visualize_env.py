
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import logging

from utils import Statistics

logger = logging.getLogger(__name__)

def evaluate_threshold(errors, labels, threshold):
    stats = Statistics()
    skip = 0
    for e, l in zip(errors, labels):
        # if len(e) < 5:
        #     skip+=1
        #     continue
        if l==1:
            if np.max(e) > threshold:
                stats.add_tp()
            else:
                stats.add_fn()
        else:
            if np.max(e) > threshold:
                stats.add_fp()
            else:
                stats.add_tn()
    # print('skippend',skip)
    return stats


def show_test_results(test, title, folder):

    logger.info(folder.parent / test['path'])
    logger.info(test['stats'].as_string())
    try:
        test_results = torch.load(folder.parent / test['path'])
    except Exception as e:
        logger.info(str(e))
        return

    errors = test_results['e']
    labels = test_results['labels']
    logger.info('spliting blocks')
    if 'block_split' in test_results:
        blocks = test_results['block_split']
    else:
        blocks = []
        for i in range(1,len(labels)):
            if labels[i] != labels[i-1]:
                blocks.append(i)
    errors = np.array(np.split(errors, blocks))
    labels = np.array(list(map(np.max,np.split(labels, blocks)))) == 1

    anomal = np.concatenate(errors[labels]).ravel()
    normal = np.concatenate(errors[~labels]).ravel()

    # label_blocks = np.array(np.split(test_results['labels'], blocks))
    # logger.info(label_blocks.shape)
    # label_blocks = label_blocks[labels]
    # logger.info(label_blocks.shape)
    # label_blocks = np.sum(label_blocks, axis=1)
    # logger.info(label_blocks.shape)
    # logger.info(f'anomal len: mean={np.mean(label_blocks)} median={np.median(label_blocks)} min={np.min(label_blocks)} max={np.max(label_blocks)}')

    plt.figure()
    plt.hist([anomal, normal], bins=20, color=['r', 'b'], label=['anomalous', 'normal'], log=True)
    plt.legend()
    # plt.xlabel('Estimated probability of being anomaly')
    plt.xlabel('Prediction error (MSE)')
    plt.ylabel('Number of logs')
    plt.savefig(folder / f'{title}_hist.png')

    p99 = np.percentile(normal,99)
    logger.info(f'Normal: mean={np.mean(normal)} median={np.median(normal)} max={np.max(normal)} p99={p99}')
    logger.info(f'Anomal: mean={np.mean(anomal)} median={np.median(anomal)} max={np.max(anomal)}')

    lengths = list(map(len, errors))
    logger.info(f'len: mean={np.mean(lengths)} median={np.median(lengths)} max={np.max(lengths)}')

    if True:
        points = 100
        step = max(test_results['e'])/points
        thresholds = []
        for i in range(points):
            logger.info(f'evaluating thresholds ({i}/{points})')
            t = i*step
            thresholds.append((t, evaluate_threshold(errors,labels,t)))
    else:
        thresholds = test_results['thresholds'][:-1]

    threshold = []
    prec = []
    recall = []
    f1 = []
    best_f = 0
    best_t = 0
    best_s = None
    for t, stats in thresholds:
        threshold.append(t)
        stats = stats.as_dict()
        prec.append(stats['precision'])
        recall.append(stats['recall'])
        f1.append(stats['f1'])
        if best_f <= stats['f1']:
            best_f = stats['f1']
            best_t = t
            best_s = stats
    logger.info(best_s)
    plt.figure()
    plt.plot(threshold, prec, 'b', label='precision')
    plt.plot(threshold, recall, 'r', label='recall')
    plt.plot(threshold, f1, 'g', label='F1')
    plt.vlines(best_t, 0,1, linestyles='dotted', label=f'threshold best ({best_t:4f}, f1={best_f:4f})')
    if 'threshold' in test_results:
        plt.vlines(test_results['threshold'], 0,1, label=f"threshold used ({test_results['threshold']:4f}, f1={test_results['stats'].as_dict()['f1']:4f})")
    plt.xlabel('Threshold')
    plt.legend()
    plt.savefig(folder / f'{title}_threshold.png')

    for l in range(len(labels)-1):
        if labels[l]!=labels[l+1]:
            i = blocks[l]
            break
    start = i-500
    limit = i+500
    block_borders = list(map(lambda x: x - start, filter(lambda x: x > start and x < limit, test_results['block_split'])))
    err_scale = max(test_results['e'][start:limit])*1.05

    plt.figure()
    plt.plot(test_results['e'][start:limit], 'r', label='prediction error')
    plt.xlabel('Time')
    plt.ylabel('Prediction error (MSE)')

    colors = [test_results['labels'][start+i+1] for i in block_borders[:-1]]
    colors = list(map(lambda x: (1., 0.9, 0.9) if x else (1., 1., 1.), colors))
    plt.axes().pcolorfast(block_borders, [0,err_scale], [colors])

    plt.vlines(block_borders, 0, err_scale, colors='black', linestyles='dotted')

    if 'threshold' in test_results:
        plt.plot([0, limit - start], [test_results['threshold'], test_results['threshold']], label='threshold')
    if 'e_s' in test_results:
        plt.plot(test_results['e_s'][start:limit], 'g', label='smooth errors')
    if 'anomalies' in test_results:
        plt.scatter(test_results['anomalies'], [test_results['e_s'][a] for a in test_results['anomalies'] if a < limit])

    plt.legend()
    plt.savefig(folder / f'{title}_test.png')


def visualize_env(path):
    env = torch.load(path+'/env.bin')
    logger.info(env.args)
    folder = pathlib.Path(path)

    if len(env.epochs) > 1:
        train_loss = [ epoch.get('train_loss',0) for epoch in env.epochs]
        validation_loss = [ epoch.get('validation_loss',0) for epoch in env.epochs]
        epochs = range(len(env.epochs))
        plt.figure()
        plt.plot(epochs, train_loss, 'b', label='train loss')
        plt.plot(epochs, validation_loss, 'r', label='validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.savefig(folder / 'loss.png')

    for test in env.test_evaluations:
        title = f"{test['epoch']}_{test['path'].name}"
        logger.info(title)
        show_test_results(test, title, folder)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        visualize_env(arg)
    if len(sys.argv) == 2:
        plt.show()
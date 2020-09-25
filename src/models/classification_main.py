import argparse
import datetime
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

import logging

from classification import LogClassification
from data.data_loaders import LogBlocksDataset, padd_batch
from visualization.visualize_env import visualize_env

logger = logging.getLogger(__name__)

def load_args(args_list):
    """Parse command-line arguments."""
    argparser = argparse.ArgumentParser()

    LogClassification.add_arguments(argparser)

    argparser.add_argument("-v", "--verbose", action="store_true", help='print logs to console')

    argparser.add_argument("--title", type=str, default="hdfs_class", help='used when generating result directory')
    argparser.add_argument("--path", type=str, default="logs", help='directory where to save results')
    argparser.add_argument("--load", type=str, default=None, help='path to existing results to resume training')

    argparser.add_argument("--only_evaluate", action="store_true", help='do not train, only evaluate on test data')
    argparser.add_argument("--threshold", type=float, default=0.5, help='threshold for anomaly detection')
    argparser.add_argument("--label_by_block", action="store_true", help='force evaluation per window, even if labels per log are available')

    argparser.add_argument("--epochs", type=int, default=10, help='number epochs to train')
    argparser.add_argument("--batch_size", type=int, default=5, help='batch size')

    argparser.add_argument("--data", type=str, required=True, help='path to preprocessed data')

    args = argparser.parse_args(args_list)

    # define torch.device
    args.device = torch.device("cpu")
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        torch.cuda.init()

    # define log path
    if args.load:
        args.path = pathlib.Path(args.load)
    else:
        timestring = datetime.datetime.now().strftime(f"%Y%m%d_%H%M%S")
        folder_name = "{}_{}".format(timestring, args.title)
        args.path = pathlib.Path(args.path) / folder_name

    args.data = pathlib.Path(args.data)

    return args


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    args = load_args(sys.argv[1:])

    if args.load:
        logger.info(f"Loading model from {args.path}")
        interface = LogClassification(path=args.path)
    else:
        logger.info(f"Creating new model with arguments\n{args}")
        interface = LogClassification(args=args)

    logger.info(f"Loading data {args.data}")
    data = torch.load(args.data)
    train = DataLoader(
        LogBlocksDataset(data['train'], transform=interface.transform),
        batch_size=args.batch_size, shuffle=True, collate_fn=padd_batch
    )
    validation = DataLoader(
        LogBlocksDataset(data['validation'], transform=interface.transform),
        batch_size=args.batch_size, shuffle=True, collate_fn=padd_batch
    )
    test = DataLoader(
        LogBlocksDataset(data['test'], transform=interface.transform),
        batch_size=args.batch_size, shuffle=True, collate_fn=padd_batch
    )
    logger.info(f"Dataset batches: train={len(train)}, validation={len(validation)}, test={len(test)}")
    label_by_block = 'labels' not in data['test'][0] or args.label_by_block

    epoch = len(interface.env.epochs)
    if not args.only_evaluate:
        logger.info("Training")
        interface.transform.compute_normalization(data['train'])
        best_validation = float('inf')
        for epoch in range(1, args.epochs + 1):
            train_loss = interface.train(train)
            logger.info(
                " | ".join(
                    [
                        "Epoch: [{:03d}/{:03d}]".format(epoch, args.epochs),
                        "Train loss: {:.3f}".format(train_loss)
                    ]
                )
            )
            message, data = interface.evaluate(validation, args.threshold, label_by_block=label_by_block)
            logger.info(message)
            data['train_loss'] = train_loss
            interface.env.epochs.append(data)
            if not interface.env.best_epoch or best_validation > data['validation_loss']:
                best_validation = data['validation_loss']
                interface.env.best_epoch = epoch
            interface.save(args.path, model=False)
            interface.save(args.path / 'epochs' / str(epoch), env=False)

        logger.info("Saving model")
        interface.save(args.path)

    logger.info("Evaluate on test data")
    result = interface.test(test, args.threshold, label_by_block=label_by_block)
    interface.env.add_test_result(epoch, args.data.name, result)
    interface.save(args.path, model=False)
    logger.info('last epoch' + result['stats'].as_string())

    logger.info('Loading best epoch')
    epoch = interface.env.best_epoch
    interface._load_model(args.path / 'epochs' / str(epoch))

    logger.info("Evaluate on test data")
    result = interface.test(test, args.threshold, label_by_block=label_by_block)
    interface.env.add_test_result(epoch, args.data.name, result)
    interface.save(args.path, model=False)
    logger.info('best epoch' + result['stats'].as_string())

    try:
        logger.info('Generating figures')
        visualize_env(str(args.path))
    except Exception as e:
        logger.info(str(e))

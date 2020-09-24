import argparse
import datetime
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

import logging

from data.data_loaders import LogBlocksDataset, padd_batch
from prediction import LogAnomalyDetection
from visualization.visualize_env import visualize_env

logger = logging.getLogger(__name__)

def load_args(args_list):
    """Parse command-line arguments."""
    argparser = argparse.ArgumentParser()

    LogAnomalyDetection.add_arguments(argparser)

    argparser.add_argument("-v", "--verbose", action="store_true", help='print logs to console')

    argparser.add_argument("--title", type=str, default="hdfs", help='used when generating result directory')
    argparser.add_argument("--path", type=str, default="logs", help='directory where to save results')
    argparser.add_argument("--load", type=str, default=None, help='path to existing results to resume training')

    argparser.add_argument("--only_evaluate", action="store_true", help='do not train, only evaluate on test data')
    argparser.add_argument("--evaluate_best", action="store_true", help='evaluate on best epoch (default is last)')
    argparser.add_argument("--label_by_block", action="store_true", help='force evaluation per window, even if labels per log are available')

    argparser.add_argument("--epochs", type=int, default=10, help='number epochs to train')
    argparser.add_argument("--batch_size", type=int, default=5, help='batch size')

    argparser.add_argument("--data", type=str, required=True, help='path to preprocessed data')
    argparser.add_argument("--limit_train", type=int, default=None, help='limit number of train windows')
    argparser.add_argument("--limit_validation", type=int, default=None, help='limit number of validation windows')

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
        prefix = 'fasttext-skipgram-'
        prefix = args.fasttext.find(prefix) + len(prefix)
        folder_name = f"{timestring}_{args.title}_detect_{args.loss}_{args.fasttext[prefix:-4]}"
        args.path = pathlib.Path(args.path) / folder_name

    return args


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    args = load_args(sys.argv[1:])
    if not args.data and not args.load:
        logger.info("ERROR missing data source, set data or load argument!")
        exit(1)

    if args.load:
        logger.info(f"Loading model from {args.path}")
        interface = LogAnomalyDetection(args, path=args.path)
        if not args.data:
            setattr(args, 'data', interface.env.args.data)
    else:
        logger.info(f"Creating new model with arguments\n{args}")
        interface = LogAnomalyDetection(args)

    logger.info(f"Loading data {args.data}")
    data = torch.load(args.data)

    label_by_block = 'labels' not in data['test'][0] or args.label_by_block
    interface.transform.label_by_block = label_by_block

    if not args.only_evaluate:
        if args.limit_train:
            data['train'] = data['train'][:args.limit_train]
        train = DataLoader(
            LogBlocksDataset(data['train'], transform=interface.transform),
            batch_size=args.batch_size, shuffle=True, collate_fn=padd_batch
        )
        if args.limit_validation:
            data['validation'] = data['validation'][:args.limit_validation]
        validation = DataLoader(
            LogBlocksDataset(data['validation'], transform=interface.transform),
            batch_size=args.batch_size, shuffle=True, collate_fn=padd_batch
        )
    else:
        train, validation = [], []

    test = DataLoader(
        LogBlocksDataset(data['test'], transform=interface.transform),
        batch_size=args.batch_size, shuffle=False, collate_fn=padd_batch
    )
    logger.info(f"Dataset batch count: train={len(train)}, validation={len(validation)}, test={len(test)}")

    if not args.only_evaluate:
        logger.info("Training")
        pretrained_epochs = len(interface.env.epochs)
        target_epochs = pretrained_epochs + args.epochs

        if pretrained_epochs == 0:
            interface.transform.compute_normalization(data['train'])
            validation_loss = interface.evaluate(validation)
            interface.env.epochs.append({'train_loss': float('inf'), 'validation_loss': validation_loss})
            logger.info("Initial validation loss: {:.3f}".format(validation_loss))

        best_loss = interface.env.epochs[-1]['validation_loss']
        for epoch in range(pretrained_epochs + 1, target_epochs + 1):
            train_loss = interface.train(train)
            validation_loss = interface.evaluate(validation)
            logger.info(" | ".join([
                "Epoch: [{:03d}/{:03d}]".format(epoch, target_epochs),
                "Train loss: {:.3f}".format(train_loss),
                "Validation loss: {:.3f}".format(validation_loss),
            ]))

            epoch_dir = f'epochs/{epoch}'
            interface.save(args.path / epoch_dir, env=False)
            interface.env.epochs.append({
                'train_loss': train_loss,
                'validation_loss': validation_loss,
                'epoch_dir': epoch_dir
            })
            if best_loss > validation_loss:
                best_loss = validation_loss
                interface.env.best_epoch = epoch
            interface.save(args.path, model=False)

    logger.info("Evaluation anomaly detection model")
    result = interface.test_detection(test, label_by_block=label_by_block, dataset_name=pathlib.Path(args.data).name)
    interface.save(args.path, model=False)
    logger.info('last epoch' + result['stats'].as_string())

    if args.evaluate_best:
        logger.info(f"Loading best model")
        interface._load_model(epoch='best')
        logger.info("Evaluation best anomaly detection model")
        result = interface.test_detection(test, label_by_block=label_by_block, dataset_name=pathlib.Path(args.data).name)
        interface.save(args.path, model=False)
        logger.info('best epoch' + result['stats'].as_string())

    try:
        logger.info('Generating figures')
        visualize_env(str(args.path))
    except Exception as e:
        logger.info(e)

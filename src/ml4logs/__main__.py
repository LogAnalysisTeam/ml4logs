# ===== IMPORTS =====
# === Standard library ===
import argparse
import logging
import pathlib
import json

# === Local ===
import ml4logs


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== MAIN =====
def main():
    ml4logs.utils.setup_logging()

    COMMANDS = {
        'download': ml4logs.data.utils.download,
        'extract': ml4logs.data.utils.extract,
        'prepare': ml4logs.data.prepare.prepare,
        'head': ml4logs.data.utils.head,
        'merge_features': ml4logs.data.utils.merge_features,
        'train_fasttext': ml4logs.features.fasttext.train_fasttext,
        'preprocess_fasttext': ml4logs.features.fasttext.preprocess_fasttext,
        'parse_ibm_drain': ml4logs.features.parser.parse_ibm_drain,
        'aggregate_by_blocks': ml4logs.features.aggregate.aggregate_by_blocks,
        'aggregate_by_lines': ml4logs.features.aggregate.aggregate_by_lines,
        'train_test_models': ml4logs.models.train_test.train_test_models,
        'train_test_seq2seq': ml4logs.models.baselines.train_test_seq2seq,
        'train_test_seq2label': ml4logs.models.baselines.train_test_seq2label
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=pathlib.Path)
    args = parser.parse_args()

    logger.info('Read config file \'%s\'', args.config_path)
    config = json.loads(args.config_path.read_text())

    logger.info('Execute pipeline')
    for step in config['pipeline']:
        if step.get('skip', False):
            logger.info('===== Skip \'%s\' step =====', step['action'])
            continue
        logger.info('===== Perform \'%s\' step =====', step['action'])
        COMMANDS[step['action']](step)


if __name__ == '__main__':
    main()

# ===== IMPORTS =====
# === Standard library ===
import logging
import pathlib

# === Thirdparty ===
import numpy as np
import fasttext

# === Local ===
import ml4logs


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def train_fasttext(args):
    logs_path = pathlib.Path(args['logs_path'])
    model_path = pathlib.Path(args['model_path'])

    ml4logs.utils.mkdirs(files=[model_path])

    logger.info('Train fasttext model on \'%s\'', logs_path)
    model = fasttext.train_unsupervised(str(logs_path), **args['model_args'])
    logger.info('Save fasttext model into \'%s\'', model_path)
    model.save_model(str(model_path))


def preprocess_fasttext(args):
    logs_path = pathlib.Path(args['logs_path'])
    model_path = pathlib.Path(args['model_path'])
    embeddings_path = pathlib.Path(args['embeddings_path'])

    ml4logs.utils.mkdirs(files=[embeddings_path])

    logger.info('Load fasttext model from \'%s\'', model_path)
    model = fasttext.load_model(str(model_path))
    n_lines = ml4logs.utils.count_file_lines(logs_path)
    step = n_lines // 10
    logger.info('Start preprocessing using fasttext')
    embeddings = []
    with logs_path.open() as logs_in_f:
        for i, line in enumerate(logs_in_f):
            embedding = model.get_sentence_vector(line.strip())
            embeddings.append(embedding)
            if i % step <= 0:
                logger.info('Processed %d / %d lines', i, n_lines)
    embeddings = np.stack(embeddings)
    logger.info('Save embeddings into \'%s\'', embeddings_path)
    np.save(embeddings_path, embeddings)

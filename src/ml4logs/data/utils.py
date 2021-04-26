# ===== IMPORTS =====
# === Standard library ===
import logging
import pathlib
import tarfile
import itertools as itools

# === Thirdparty ===
import requests
import numpy as np

# === Local ===
import ml4logs


# ===== CONSTANTS =====
CHUNK_SIZE = 8196


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def download(args):
    path = pathlib.Path(args['path'])

    if not args['force'] and path.exists():
        logger.info('File already exists and force is false')
        return

    ml4logs.utils.mkdirs(files=[path])

    logger.info('Download \'%s\'', args['url'])
    with requests.get(args['url'], stream=True) as r:
        r.raise_for_status()
        with path.open('wb') as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)


def extract(args):
    in_path = pathlib.Path(args['in_path'])
    out_dir = pathlib.Path(args['out_dir'])

    ml4logs.utils.mkdirs(folders=[out_dir])

    logger.info('Open \'%s\' as tarfile', in_path)
    with tarfile.open(in_path, 'r:gz') as tar:
        members = tar.getmembers()
        logger.info('Extract %d files', len(members))
        tar.extractall(out_dir, members=members)


def head(args):
    logs_path = pathlib.Path(args['logs_path'])
    logs_head_path = pathlib.Path(args['logs_head_path'])

    ml4logs.utils.mkdirs(files=[logs_head_path])

    logger.info('Read first %d lines from \'%s\'', args['n_rows'], logs_path)
    with logs_path.open(encoding='utf8') as in_f:
        logs_head = tuple(map(
            lambda l: l.strip(), itools.islice(in_f, args['n_rows'])))
    logger.info('Save them into \'%s\'', logs_head_path)
    logs_head_path.write_text('\n'.join(logs_head))


def merge_features(args):
    merged_path = pathlib.Path(args['merged_path'])

    logger.info('Load all input arrays')
    arrays = []
    for path_str in args['features_paths']:
        array = np.load(pathlib.Path(path_str))
        arrays.append(array)
    logger.info('Concatenate input arrays')
    merged_array = np.concatenate(arrays, axis=1)

    logger.info('Save merged array into \'%s\'', merged_path)
    np.save(merged_path, merged_array)

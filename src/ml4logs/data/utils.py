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


# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def download(args):
    path = pathlib.Path(args['path'])
    if path.exists():
        logger.warn(f"File: {path} already exists! Skipping download...")
        return

    ml4logs.utils.mkdirs(files=[path])

    logger.info('Download \'%s\'', args['url'])
    response = requests.get(args['url'])
    logger.info('Save into \'%s\'', path)
    path.write_bytes(response.content)


def extract(args):
    in_path = pathlib.Path(args['in_path'])
    out_dir = pathlib.Path(args['out_dir'])

    ml4logs.utils.mkdirs(folders=[out_dir])

    logger.info('Open \'%s\' as tarfile', in_path)
    with tarfile.open(in_path, 'r:gz') as tar:
        members = tar.getmembers()
        logger.info('Extracting %d files', len(members))
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, out_dir, members=members)


def head(args):
    logs_path = pathlib.Path(args['logs_path'])
    logs_head_path = pathlib.Path(args['logs_head_path'])

    ml4logs.utils.mkdirs(files=[logs_head_path])

    logger.info('Read first %d lines from \'%s\'', args['n_rows'], logs_path)
    with logs_path.open() as in_f:
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
        assert 0 < array.ndim <= 2
        if array.ndim == 1:
            array = array.reshape(-1, 1) 
        arrays.append(array)
    logger.info('Concatenate input arrays')
    merged_array = np.concatenate(arrays, axis=1)

    logger.info('Save merged array into \'%s\'', merged_path)
    np.save(merged_path, merged_array)

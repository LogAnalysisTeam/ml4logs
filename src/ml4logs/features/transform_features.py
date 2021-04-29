# ===== IMPORTS =====
# === Standard library ===
from collections import defaultdict, OrderedDict
import logging
import pathlib
from pathlib import Path
import re
import typing
from typing import Generator, Iterable, List, Dict

# === Thirdparty ===
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# === Local ===
import ml4logs
from ml4logs.features.utils import load_features

# ===== GLOBALS =====
logger = logging.getLogger(__name__)


def transform_features(args):
    assert args["method"] in ["scale_minmax"]

    data_dir = Path(args['data_dir'])
    ml4logs.utils.mkdirs(folders=[data_dir])

    METHODS = {
        "scale_minmax": lambda: MinMaxScaler(clip=True, copy=False),
        "scale_standardize": lambda: StandardScaler(copy=False)
    }

    trans = METHODS[args["method"]]()

    # TODO loading all features into memory - this might be problem, implement batches if needed
    features = np.vstack([load_features(Path(data_dir, f)) for f in args["fit"]])
    logger.info(f"Imported features to fit transform, shape={features.shape}")

    logger.info("Fitting the transform.")
    trans.fit(features)

    if "save_transform_path" in args:
        save_transform_path = pathlib.Path(args['save_transform_path'])
        logger.info(f'Saving "{args["method"]}" transform as "{save_transform_path}"')
        joblib.dump(trans, save_transform_path)

    logger.info('Starting actual transforms')
    for pair in args["transform"]:
        source_path = Path(data_dir, pair["source"])
        target_path = Path(data_dir, pair["target"])
        logger.info(f'Processing "{source_path}"')

        features = load_features(Path(source_path))
        transformed_features = trans.transform(features)

        logger.info(f'Saving transformed features as "{target_path}", shape={transformed_features.shape}')
        np.save(target_path, transformed_features)
    
    if args.get("remove_sources", False):
        logger.info('Removing transformation sources:')
        for pair in args["transform"]:
            source_path = Path(data_dir, pair["source"])
            logger.info(f'  "{source_path}"')
            source_path.unlink()





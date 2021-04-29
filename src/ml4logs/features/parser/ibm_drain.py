# ===== IMPORTS =====
# === Standard library ===
from collections import defaultdict
import logging
import pathlib
import re

# === Thirdparty ===
import drain3
import numpy as np
import pandas as pd

# === Local ===
import ml4logs

# ===== GLOBALS =====
logger = logging.getLogger(__name__)


# ===== FUNCTIONS =====
def parse_ibm_drain(args):
    templates_path = pathlib.Path(args['templates_path'])
    data_dir = pathlib.Path(args['data_dir'])

    ml4logs.utils.mkdirs(files=[templates_path], folders=[data_dir])

    pattern = re.compile(args['regex'])
    template_miner = drain3.TemplateMiner()
    eventids_str = defaultdict(list)

    logger.info('Starting IBM/Drain3 parser')
    for pair in args["pairs"]:
        logs_path = pathlib.Path(data_dir, pair["logs_name"])
        eventids_path = pathlib.Path(data_dir, pair["eventids_name"])
        logger.info(f'Processing: {logs_path}')

        n_lines = ml4logs.utils.count_file_lines(logs_path)
        step = n_lines // 10

        with logs_path.open() as logs_in_f:
            for i, line in enumerate(logs_in_f):
                match = pattern.fullmatch(line.strip())
                content = match.group('content')
                result = template_miner.add_log_message(content)
                eventids_str[logs_path].append(result['cluster_id'])
                if i % step <= 0:
                    logger.info('Processed %d / %d lines', i, n_lines)

    cluster_mapping = {}
    templates = []
    logger.info('Factorizing cluster ids')
    for i, cluster in enumerate(template_miner.drain.clusters):
        cluster_mapping[cluster.cluster_id] = i
        templates.append(
            [i, cluster.size, ' '.join(cluster.log_template_tokens)])


    logger.info('Save templates')
    templates_df = pd.DataFrame(
        templates, columns=['event_id', 'occurrences', 'template'])
    templates_df.to_csv(templates_path, index=False)

    for pair in args["pairs"]:
        logs_path = pathlib.Path(data_dir, pair["logs_name"])
        eventids_path = pathlib.Path(data_dir, pair["eventids_name"])
        logger.info(f'Saving eventids to {eventids_path}')
        eventids = np.array(list(map(cluster_mapping.get, eventids_str[logs_path])))
        np.save(eventids_path, eventids)
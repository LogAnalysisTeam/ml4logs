# ===== IMPORTS =====
# === Standard library ===
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
    logs_path = pathlib.Path(args['logs_path'])
    eventids_path = pathlib.Path(args['eventids_path'])
    templates_path = pathlib.Path(args['templates_path'])

    ml4logs.utils.mkdirs(files=[eventids_path, templates_path])

    pattern = re.compile(args['regex'])
    template_miner = drain3.TemplateMiner()
    eventids_str = []
    n_lines = ml4logs.utils.count_file_lines(logs_path)
    step = n_lines // 10
    logger.info('Start parsing using IBM/Drain3')
    with logs_path.open() as logs_in_f:
        for i, line in enumerate(logs_in_f):
            match = pattern.fullmatch(line.strip())
            content = match.group('content')
            result = template_miner.add_log_message(content)
            eventids_str.append(result['cluster_id'])
            if i % step <= 0:
                logger.info('Processed %d / %d lines', i, n_lines)

    cluster_mapping = {}
    templates = []
    logger.info('Factorize cluster ids')
    for i, cluster in enumerate(template_miner.drain.clusters):
        cluster_mapping[cluster.cluster_id] = i
        templates.append(
            [i, cluster.size, ' '.join(cluster.log_template_tokens)])
    eventids = np.array(list(map(cluster_mapping.get, eventids_str)))

    logger.info('Save eventids')
    np.save(eventids_path, eventids)
    logger.info('Save templates')
    templates_df = pd.DataFrame(
        templates, columns=['event_id', 'occurrences', 'template'])
    templates_df.to_csv(templates_path, index=False)

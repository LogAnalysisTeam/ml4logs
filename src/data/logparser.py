#
# Original work at: https://github.com/LogAnalysisTeam/methods4logfiles/
#
# Modified for the purposes of Contextual Embeddings for Anomaly Detection in Logs thesis
#
#

from drain3 import TemplateMiner
from collections import defaultdict
import pickle
from typing import List, Dict, DefaultDict


def get_log_templates(clusters: List) -> Dict:
    ret = {cl.cluster_id: ' '.join(cl.log_template_tokens) for cl in clusters}
    return ret


def get_log_structure(log_lines: DefaultDict, cluster_ids: DefaultDict, clusters: List) -> Dict:
    templates = get_log_templates(clusters)

    ret_log_structure = {key: [] for key in log_lines.keys()}
    for key in ret_log_structure.keys():
        for curr_id, log in zip(cluster_ids[key], log_lines[key]):
            ret_log_structure[key].append((curr_id, log, templates[curr_id]))
    return ret_log_structure


def parse_file_drain3(data: DefaultDict) -> Dict:
    template_miner = TemplateMiner()

    cluster_ids = defaultdict(list)
    log_lines = defaultdict(list)
    for block_id, logs in data.items():
        for log in logs:
            line = log.rstrip().partition(': ')[2]  # produces tuple (pre, delimiter, post)
            result = template_miner.add_log_message(line)
            cluster_ids[block_id].append(result['cluster_id'])
            log_lines[block_id].append(line)

    log_structure = get_log_structure(log_lines, cluster_ids, template_miner.drain.clusters)
    return log_structure


def save_drain3_to_file(data: Dict, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_drain3(file_path: str) -> Dict:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    from src.data.hdfs import load_data

    q = parse_file_drain3(load_data('../../data/raw/HDFS1/HDFS.log'))

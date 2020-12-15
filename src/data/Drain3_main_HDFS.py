import json
import logging
import os
import subprocess
import sys
import time
import pathlib
import pandas as pd

from drain3 import TemplateMiner
from drain3.persistence_handler import PersistenceHandler


class FilePersistence(PersistenceHandler):
    def __init__(self, file_path):
        self.file_path = file_path
        self.state = None

    def __del__(self):
        pathlib.Path(self.file_path).write_bytes(self.state)

    def save_state(self, state):
        self.state = state

    def load_state(self):
        if not os.path.exists(self.file_path):
            return None
        self.state = pathlib.Path(self.file_path).read_bytes()
        return self.state


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

input_log = sys.argv[1]
output_dir = sys.argv[2]
persistence = FilePersistence(output_dir + "/HDFS_Drain3.state.bin")
template_miner = TemplateMiner(persistence)

line_count = 0
start_time = time.time()
batch_start_time = start_time
batch_size = 10000
structure_id = []
lines = []
with open(input_log) as f:
    for line in f:
        line = line.rstrip()
        line = line.partition(": ")[2]
        result = template_miner.add_log_message(line)
        line_count += 1
        if result["change_type"] != "none":
            result_json = json.dumps(result)
            logger.info(f"Input ({line_count}): " + line)
            logger.info("Result: " + result_json)
        structure_id.append(result["cluster_id"])
        lines.append(line)

time_took = time.time() - start_time
rate = line_count / time_took
logger.info(f"--- Done processing file. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
            f"{len(template_miner.drain.clusters)} clusters")

sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)

templates_out = []
for cluster in sorted_clusters:
    templates_out.append([cluster.cluster_id, cluster.size, ' '.join(cluster.log_template_tokens)])
pd.DataFrame(templates_out, columns=["EventId", "Occurrences", "EventTemplate"]). \
    to_csv(output_dir + "/HDFS_Drain3.log_templates.csv", index=False)

structured_out = []
for i, item in enumerate(structure_id):
    for cluster in sorted_clusters:
        if item == cluster.cluster_id:
            structured_out.append([item, lines[i], ' '.join(cluster.log_template_tokens)])

pd.DataFrame(structured_out, columns=["EventId", "Content", "EventTemplate"]). \
    to_csv(output_dir + "/HDFS_Drain3.log_structured.csv", index=False)

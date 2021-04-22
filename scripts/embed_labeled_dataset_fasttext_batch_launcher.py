import subprocess, os
import re
import json
from pathlib import Path

if __name__ == '__main__':
    models_paths = [
        '/home/cernypro/dev/source/ml4logs/models/embeddings/fasttext_from_no_timestamps_train-data-HDFS1-cv1-1-time-ordered.bin'
    ]

    datasets_paths = [
        ('/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/no_timestamps_test-data-HDFS1.log', '/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/test-labels-HDFS1.csv'),
        ('/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/no_timestamps_train-data-HDFS1-cv1-1.log','/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/train-labels-HDFS1-cv1-1.csv'),
        ('/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/no_timestamps_val-data-HDFS1-cv1-1.log','/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/val-labels-HDFS1-cv1-1.csv')
    ]
    SUB_PAT = re.compile(r'Submitted batch job (\d+)')
    jobs_details = {}
    for model_path in models_paths:
        job_nums = []
        for dataset, labels in datasets_paths:
            my_env = os.environ.copy()
            my_env['MODEL_PATH'] = model_path
            my_env['LOGFILE_PATH'] = dataset
            my_env['LABEL_CSV_PATH'] = labels
            # subprocess.run(['sbatch', 'embed_labeled_dataset_env_script.batch'], env=my_env)
            ret = subprocess.run(['sbatch', 'embed_labeled_dataset_fasttext_env_script.batch'], env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            job_nums.append(SUB_PAT.match(ret.stdout.decode('utf-8')).group(1))
        jobs_details[Path(model_path).stem] = '--dependency=afterok:' + ','.join(job_nums)

    print(json.dumps(jobs_details, indent=4))
import subprocess, os
from pathlib import Path
import re

SUBMITTED_TASK_PATTERN = re.compile(r'Submitted batch job (\d+)')

def run_train_ict_from_preprocessed(two_tower: bool,
                                    encoder_type: str,
                                    bert_model: str,
                                    dataset_run_name: str,
                                    dataset_path,
                                    eval_dataset_path,
                                    output_encode_dim: int=100,
                                    target_max_seq_len: int=512,
                                    eval_steps: int=7500,
                                    save_steps: int=5000,
                                    logging_steps: int=100,
                                    ):
    my_env = os.environ.copy()
    my_env['TWO_TOWER'] = str(int(two_tower))
    my_env['ENCODER_TYPE'] = encoder_type
    my_env['BERT_MODEL'] = bert_model
    my_env['OUTPUT_ENCODE_DIM'] = str(output_encode_dim)
    my_env['TRAIN_BATCH_SIZE'] = str(train_batch_size)
    my_env['TARGET_MAX_SEQ_LEN'] = str(target_max_seq_len)
    my_env['EVAL_STEPS'] = str(eval_steps)
    my_env['SAVE_STEPS'] = str(save_steps)
    my_env['LOGGING_STEPS'] = str(logging_steps)

    my_env['DATASET_NAME'] = dataset_run_name
    my_env['TRAIN_DATASET'] = str(dataset_path)
    my_env['EVAL_DATASET'] = str(eval_dataset_path)
    
    ret = subprocess.run(['sbatch', 'ict_pretrain_preprocessed_env_script.batch'], env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    job_num = int(SUBMITTED_TASK_PATTERN.match(ret.stdout.decode('utf-8')).group(1))
    return job_num
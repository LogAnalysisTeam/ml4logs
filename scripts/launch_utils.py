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
                                    train_batch_size: int=32,
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


def embed_labeled_datasets_using_bert_model(model_path,
                                            datasets_paths,
                                            dependencies = None):
    command_list = ['sbatch']
    if dependencies:
        command_list.append('--dependency=afterok:' + ','.join(map(str, dependencies)))
    command_list.append('embed_labeled_dataset_env_script.batch')
    job_nums = []
    for dataset, labels in datasets_paths:
        my_env = os.environ.copy()
        my_env['MODEL_PATH'] = str(model_path)
        my_env['LOGFILE_PATH'] = str(dataset)
        my_env['LABEL_CSV_PATH'] = str(labels)
        ret = subprocess.run(command_list, env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        job_nums.append(int(SUBMITTED_TASK_PATTERN.match(ret.stdout.decode('utf-8')).group(1)))
    return job_nums


def launch_aetcn_experiment(model_name:str,
                            train_basepath: Path,
                            val_basepath: Path,
                            test_basepath: Path,
                            dependencies=None):
    command_list = ['sbatch']
    if dependencies:
        command_list.append('--dependency=afterok:' + ','.join(map(str, dependencies)))
    command_list.append('aetcn_experiment_env.batch')

    my_env = os.environ.copy()
    embedding_folder_name = f'labeled_embedding_from_{model_name}'
    train_dir = train_basepath / embedding_folder_name
    val_dir = val_basepath / embedding_folder_name
    test_dir = test_basepath / embedding_folder_name
    my_env['OUTPUT_PARENT_FOLDER'] = '/home/cernypro/dev/source/ml4logs/models/anomaly_detection'
    my_env['EMBEDDING_MODEL_NAME'] = model_name
    my_env['X_TRAIN_DATASET'] = str(train_dir / 'X.pickle')
    my_env['Y_TRAIN_DATASET'] = str(train_dir / 'y.npy')
    my_env['X_VAL_DATASET'] = str(val_dir / 'X.pickle')
    my_env['Y_VAL_DATASET'] = str(val_dir / 'y.npy')
    my_env['X_TEST_DATASET'] = str(test_dir / 'X.pickle')
    my_env['Y_TEST_DATASET'] = str(test_dir / 'y.npy')
    
    ret = subprocess.run(command_list, env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    job_num = int(SUBMITTED_TASK_PATTERN.match(ret.stdout.decode('utf-8')).group(1))
    return job_num
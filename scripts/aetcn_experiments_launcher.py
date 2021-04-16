import subprocess, os
from pathlib import Path

if __name__ == '__main__':
    model_names = [
        'LogEncoder_from_1T__Eps_1_Combined_20210401_uniform_3_epochs_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
        'LogEncoder_from_1T_Eps_1_Combined_20210401_max-avg_3_epochs_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
        'LogEncoder_from_1T_Eps_1_Combined_20210401_uniform_3_epochs_Seed-42_T-len_170_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100',
        'LogEncoder_from_2T__Eps_1_Combined_20210401_uniform_3_epochs_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
        'LogEncoder_from_2T_Eps_1_Combined_20210401_max-avg_3_epochs_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100'
    ]

    train_basepath = Path('/home/cernypro/dev/source/ml4logs/data/interim/no_timestamps_train-data-HDFS1-cv1-1')
    val_basepath = Path('/home/cernypro/dev/source/ml4logs/data/interim/no_timestamps_val-data-HDFS1-cv1-1')
    test_basepath = Path('/home/cernypro/dev/source/ml4logs/data/interim/no_timestamps_test-data-HDFS1')

    for model_name in model_names:
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
        
        subprocess.run(['sbatch', 'aetcn_experiment_env.batch'], env=my_env)
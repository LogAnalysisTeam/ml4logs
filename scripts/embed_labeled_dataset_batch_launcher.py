import subprocess, os

if __name__ == '__main__':
    models_paths = [
        '/home/cernypro/dev/source/ml4logs/models/LogEncoder_from_1T__Eps_1_Combined_20210401_uniform_3_epochs_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
        '/home/cernypro/dev/source/ml4logs/models/LogEncoder_from_1T_Eps_1_Combined_20210401_max-avg_3_epochs_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
        '/home/cernypro/dev/source/ml4logs/models/LogEncoder_from_1T_Eps_1_Combined_20210401_uniform_3_epochs_Seed-42_T-len_170_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100',
        '/home/cernypro/dev/source/ml4logs/models/LogEncoder_from_2T__Eps_1_Combined_20210401_uniform_3_epochs_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
        '/home/cernypro/dev/source/ml4logs/models/LogEncoder_from_2T_Eps_1_Combined_20210401_max-avg_3_epochs_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100'
    ]

    datasets_paths = [
        ('/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/no_timestamps_test-data-HDFS1.log', '/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/test-labels-HDFS1.csv'),
        ('/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/no_timestamps_train-data-HDFS1-cv1-1.log','/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/train-labels-HDFS1-cv1-1.csv'),
        ('/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/no_timestamps_val-data-HDFS1-cv1-1.log','/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/val-labels-HDFS1-cv1-1.csv')
    ]

    for model_path in models_paths:
        for dataset, labels in datasets_paths:
            my_env = os.environ.copy()
            my_env['MODEL_PATH'] = model_path
            my_env['LOGFILE_PATH'] = dataset
            my_env['LABEL_CSV_PATH'] = labels
            subprocess.run(['sbatch', 'embed_labeled_dataset_env_script.batch'], env=my_env)
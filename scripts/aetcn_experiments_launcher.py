import subprocess, os
from pathlib import Path

if __name__ == '__main__':
    # model_names = [
    #     'LogEncoder_from_1T__Eps_1_Combined_20210401_uniform_3_epochs_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_1T_Eps_1_Combined_20210401_max-avg_3_epochs_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_1T_Eps_1_Combined_20210401_uniform_3_epochs_Seed-42_T-len_170_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_2T__Eps_1_Combined_20210401_uniform_3_epochs_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_2T_Eps_1_Combined_20210401_max-avg_3_epochs_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100'
    # ]

    # model_names = [
    #     'LogEncoder_from_1T_Eps_1_M_basic_chunked_10_Seed-42_T-len_512_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_1T_Eps_1_M_chunked_by_blocks_min3_max10_Seed-42_T-len_512_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_1T_Eps_1_M_time_ordered_chunked_10_Seed-42_T-len_512_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_2T_Eps_1_M_basic_chunked_10_Seed-42_T-len_512_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_2T_Eps_1_M_chunked_by_blocks_min3_max10_Seed-42_T-len_512_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_2T_Eps_1_M_time_ordered_chunked_10_Seed-42_T-len_512_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_1T_Eps_1_Combined_20210401_concat-to-max_epochs-5_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_1T_Eps_1_Combined_20210401_smart-average_epochs-5_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_2T_Eps_1_Combined_20210401_concat-to-max_epochs-5_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_2T_Eps_1_Combined_20210401_smart-average_epochs-5_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_1T_Eps_1_Combined_20210401_concat-to-max_epochs-3_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_1T_Eps_1_Combined_20210401_smart-average_epochs-3_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_2T_Eps_1_Combined_20210401_concat-to-max_epochs-3_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100',
    #     'LogEncoder_from_2T_Eps_1_Combined_20210401_smart-average_epochs-3_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100'
    # ]

    # model_names = [
    #     "fasttext_from_no_timestamps_train-data-HDFS1-cv1-1-time-ordered"
    # ]

    # model_deps = {
    #     "LogEncoder_from_1T_Eps_1_M_basic_chunked_10_Seed-42_T-len_512_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100": "--dependency=afterok:2317355,2317356,2317357",
    #     "LogEncoder_from_1T_Eps_1_M_chunked_by_blocks_min3_max10_Seed-42_T-len_512_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100": "--dependency=afterok:2317358,2317359,2317360",
    #     "LogEncoder_from_1T_Eps_1_M_time_ordered_chunked_10_Seed-42_T-len_512_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100": "--dependency=afterok:2317361,2317362,2317363",
    #     "LogEncoder_from_2T_Eps_1_M_basic_chunked_10_Seed-42_T-len_512_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100": "--dependency=afterok:2317364,2317365,2317366",
    #     "LogEncoder_from_2T_Eps_1_M_chunked_by_blocks_min3_max10_Seed-42_T-len_512_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100": "--dependency=afterok:2317367,2317368,2317369",
    #     "LogEncoder_from_2T_Eps_1_M_time_ordered_chunked_10_Seed-42_T-len_512_C-len_512_Tr-batch_64_Ev-b_64_O-dim_100": "--dependency=afterok:2317370,2317371,2317372",
    #     "LogEncoder_from_1T_Eps_1_Combined_20210401_concat-to-max_epochs-5_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100": "--dependency=afterok:2317373,2317374,2317375",
    #     "LogEncoder_from_1T_Eps_1_Combined_20210401_smart-average_epochs-5_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100": "--dependency=afterok:2317376,2317377,2317378",
    #     "LogEncoder_from_2T_Eps_1_Combined_20210401_concat-to-max_epochs-5_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100": "--dependency=afterok:2317379,2317380,2317381",
    #     "LogEncoder_from_2T_Eps_1_Combined_20210401_smart-average_epochs-5_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100": "--dependency=afterok:2317382,2317383,2317384",
    #     "LogEncoder_from_1T_Eps_1_Combined_20210401_concat-to-max_epochs-3_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100": "--dependency=afterok:2317385,2317386,2317387",
    #     "LogEncoder_from_1T_Eps_1_Combined_20210401_smart-average_epochs-3_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100": "--dependency=afterok:2317388,2317389,2317390",
    #     "LogEncoder_from_2T_Eps_1_Combined_20210401_concat-to-max_epochs-3_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100": "--dependency=afterok:2317391,2317392,2317393",
    #     "LogEncoder_from_2T_Eps_1_Combined_20210401_smart-average_epochs-3_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100": "--dependency=afterok:2317394,2317395,2317396"
    # }

    # model_deps = {
    #     "fasttext_from_no_timestamps_train-data-HDFS1-cv1-1-time-ordered": "--dependency=afterok:2321473,2321474,2321475"
    # }

    model_deps = {
        "LogEncoder_from_RobertaCls_distilroberta-base_1T_Combined_20210401_roberta_concat-to-max_epochs-3_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100": "--dependency=afterok:2321657,2321658,2321659",
        "LogEncoder_from_RobertaCls_distilroberta-base_1T_Combined_20210401_roberta_smart-average_epochs-3_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100": "--dependency=afterok:2321660,2321661,2321662",
        "LogEncoder_from_RobertaCls_distilroberta-base_2T_Combined_20210401_roberta_concat-to-max_epochs-3_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100": "--dependency=afterok:2321663,2321664,2321665",
        "LogEncoder_from_RobertaCls_distilroberta-base_2T_Combined_20210401_roberta_smart-average_epochs-3_seed-43_Seed-42_T-len_512_C-len_512_Tr-batch_32_Ev-b_64_O-dim_100": "--dependency=afterok:2321666,2321667,2321668"
    }

    model_names = list(model_deps.keys())

    train_basepath = Path('/home/cernypro/dev/source/ml4logs/data/interim/no_timestamps_train-data-HDFS1-cv1-1')
    val_basepath = Path('/home/cernypro/dev/source/ml4logs/data/interim/no_timestamps_val-data-HDFS1-cv1-1')
    test_basepath = Path('/home/cernypro/dev/source/ml4logs/data/interim/no_timestamps_test-data-HDFS1')

    # dependencies = '--dependency=afterok:' + ','.join(list(map(str,range(2314963,2314963+12))))

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
        
        subprocess.run(['sbatch', model_deps[model_name], 'aetcn_experiment_env.batch'], env=my_env)
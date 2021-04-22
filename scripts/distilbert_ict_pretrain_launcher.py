import subprocess, os
from pathlib import Path

if __name__ == '__main__':

    datasets_basepath = Path('/home/cernypro/dev/source/ml4logs/data/processed/Combined_20210401_train_only_target_flat_context')

    epochs = [3]
    seeds = [43]
    truncations = ['Smart_Average', 'Concat_To_Max']

    for epoch in epochs:
        for seed in seeds:
            for truncation in truncations:
                dataset_folder_name = f'flattened_contexts_epochs-{epoch}_seed-{seed}_truncation-{truncation}'
                dataset_path = datasets_basepath / dataset_folder_name
                dataset_run_name = f'Combined_20210401_{truncation.lower().replace("_", "-")}_epochs-{epoch}_seed-{seed}'
                for two_tower in [0, 1]:
                    my_env = os.environ.copy()
                    my_env['TWO_TOWER'] = str(int(two_tower))
                    my_env['OUTPUT_ENCODE_DIM'] = str(100)
                    my_env['TRAIN_BATCH_SIZE'] = str(32)
                    my_env['TARGET_MAX_SEQ_LEN'] = str(512)
                    my_env['EVAL_STEPS'] = str(7500)
                    my_env['SAVE_STEPS'] = str(5000)
                    my_env['LOGGING_STEPS'] = str(50)

                    my_env['DATASET_NAME'] = dataset_run_name
                    my_env['TRAIN_DATASET'] = str(dataset_path)
                    my_env['EVAL_DATASET'] = '/home/cernypro/dev/source/ml4logs/data/processed/val-data-HDFS1-cv1-1-time-ordered_Epochs-1_Seed-43'
                    
                    subprocess.run(['sbatch', 'distilbert_ict_pretrain_preprocessed_env_script.batch'], env=my_env)
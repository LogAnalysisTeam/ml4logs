import subprocess, os
from pathlib import Path

from launch_utils import run_train_ict_from_preprocessed

if __name__ == '__main__':
    datasets_basepath = Path('/home/cernypro/dev/source/ml4logs/data/processed/Combined_20210401_roberta_only_target_flat_context')

    epochs = [3]
    seeds = [43]
    truncations = ['Smart_Average', 'Concat_To_Max']

    for epoch in epochs:
        for seed in seeds:
            for truncation in truncations:
                dataset_folder_name = f'flattened_contexts_epochs-{epoch}_seed-{seed}_truncation-{truncation}'
                dataset_path = datasets_basepath / dataset_folder_name
                dataset_run_name = f'Combined_20210401_roberta_{truncation.lower().replace("_", "-")}_epochs-{epoch}_seed-{seed}'
                for two_tower in [False, True]:
                    run_train_ict_from_preprocessed(two_tower=two_tower,
                                                    encoder_type='RobertaCls',
                                                    bert_model='distilroberta-base',
                                                    dataset_run_name=dataset_run_name,
                                                    dataset_path=dataset_path,
                                                    eval_dataset_path='/home/cernypro/dev/source/ml4logs/data/processed/val-data-HDFS1-cv1-1-time-ordered_chunked-10_roberta/flattened_contexts_epochs-1_seed-43_truncation-Concat_To_Max',
                                                    output_encode_dim=100,
                                                    target_max_seq_len=512,
                                                    eval_steps=7500,
                                                    save_steps=5000,
                                                    logging_steps=100)
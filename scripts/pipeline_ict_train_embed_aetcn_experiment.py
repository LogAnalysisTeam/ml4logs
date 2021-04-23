import subprocess, os
from pathlib import Path

from launch_utils import run_train_ict_from_preprocessed, embed_labeled_datasets_using_bert_model, launch_aetcn_experiment

def get_run_name(two_tower: bool,
                 fp16: bool,
                 dataset_name: str,
                 seed: int,
                 target_max_seq_len: int,
                 context_max_seq_len: int,
                 train_batch_size: int,
                 eval_batch_size: int,
                 output_encode_dim: int,
                 encoder_type: str,
                 pretrained_checkpoint: str):
    return f'{encoder_type} {pretrained_checkpoint} {"2T" if two_tower else "1T"}{" fp16" if fp16 else ""} {dataset_name} Seed-{seed} T-len {target_max_seq_len} C-len {context_max_seq_len} Tr-batch {train_batch_size} Ev-b {eval_batch_size} O-dim {output_encode_dim}'

if __name__ == '__main__':
    ict_datasets_basepath = Path('/home/cernypro/dev/source/ml4logs/data/processed/Combined_20210401_roberta_only_target_flat_context')

    datasets_to_embed_paths = [
        ('/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/no_timestamps_test-data-HDFS1.log', '/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/test-labels-HDFS1.csv'),
        ('/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/no_timestamps_train-data-HDFS1-cv1-1.log','/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/train-labels-HDFS1-cv1-1.csv'),
        ('/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/no_timestamps_val-data-HDFS1-cv1-1.log','/home/cernypro/dev/source/ml4logs/data/interim/HDFS1/val-labels-HDFS1-cv1-1.csv')
    ]

    aetcn_train_basepath = Path('/home/cernypro/dev/source/ml4logs/data/interim/no_timestamps_train-data-HDFS1-cv1-1')
    aetcn_val_basepath = Path('/home/cernypro/dev/source/ml4logs/data/interim/no_timestamps_val-data-HDFS1-cv1-1')
    aetcn_test_basepath = Path('/home/cernypro/dev/source/ml4logs/data/interim/no_timestamps_test-data-HDFS1')

    epochs = [3]
    seeds = [43]
    truncations = ['Smart_Average', 'Concat_To_Max']

    # ict_datasets_full_paths = []

    ict_datasets_full_paths = [
        (Path('/home/cernypro/dev/source/ml4logs/data/processed/train-data-HDFS1-cv1-1_roberta/flattened_contexts_epochs-4_seed-43_truncation-Concat_To_Max'), 'M_basic_chunked_10_roberta'),
        (Path('/home/cernypro/dev/source/ml4logs/data/processed/train-data-HDFS1-cv1-1-time-ordered_roberta/flattened_contexts_epochs-4_seed-43_truncation-Concat_To_Max'), 'M_time_ordered_chunked_10_roberta')
    ]

    for epoch in epochs:
        for seed in seeds:
            for truncation in truncations:
                dataset_folder_name = f'flattened_contexts_epochs-{epoch}_seed-{seed}_truncation-{truncation}'
                dataset_path = ict_datasets_basepath / dataset_folder_name
                dataset_run_name = f'Combined_20210401_roberta_{truncation.lower().replace("_", "-")}_epochs-{epoch}_seed-{seed}'
                ict_datasets_full_paths.append((dataset_path, dataset_run_name))
    
    for dataset_path, dataset_run_name in ict_datasets_full_paths:
        for two_tower in [False, True]:
            target_max_seq_len = 512
            train_batch_size = 32
            output_encode_dim = 100
            encoder_type = 'RobertaMean'
            bert_model = 'distilroberta-base'
            ict_job_num = run_train_ict_from_preprocessed(two_tower=two_tower,
                                                        encoder_type=encoder_type,
                                                        bert_model=bert_model,
                                                        dataset_run_name=dataset_run_name,
                                                        dataset_path=dataset_path,
                                                        eval_dataset_path='/home/cernypro/dev/source/ml4logs/data/processed/val-data-HDFS1-cv1-1-time-ordered_chunked-10_roberta/flattened_contexts_epochs-1_seed-43_truncation-Concat_To_Max',
                                                        output_encode_dim=output_encode_dim,
                                                        train_batch_size=train_batch_size,
                                                        target_max_seq_len=target_max_seq_len,
                                                        eval_steps=7500,
                                                        save_steps=5000,
                                                        logging_steps=100)
            ict_name = get_run_name(two_tower=two_tower,
                                    fp16=False,
                                    dataset_name=dataset_run_name,
                                    seed=42,
                                    target_max_seq_len=target_max_seq_len,
                                    context_max_seq_len=512,
                                    train_batch_size=train_batch_size,
                                    eval_batch_size=64,
                                    output_encode_dim=output_encode_dim,
                                    encoder_type=encoder_type,
                                    pretrained_checkpoint=bert_model).replace(' ', '_')
            print(f"{ict_name}: Job {ict_job_num}")

            log_encoder_name = f'LogEncoder_from_{ict_name}'
            log_encoder_path = Path('/home/cernypro/dev/source/ml4logs/models/encoders') / log_encoder_name
            embed_job_nums = embed_labeled_datasets_using_bert_model(log_encoder_path, datasets_to_embed_paths, dependencies=[ict_job_num])
            print(f'{log_encoder_name}: Jobs {",".join(map(str, embed_job_nums))}')

            aetcn_job_num = launch_aetcn_experiment(log_encoder_name, aetcn_train_basepath, aetcn_val_basepath, aetcn_test_basepath, dependencies=embed_job_nums)
            print(f'AETCN {log_encoder_name}: Job {aetcn_job_num}')



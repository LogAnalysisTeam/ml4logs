from datasets import load_dataset, load_from_disk
from pathlib import Path
from typing import List
import numpy as np
import time

from dataset_utils import prepare_ict

def compute_train_eval_sizes(desired_context_count, dataset_context_count, train_batch_size, eval_batch_size, eval_percentage=0.2, eval_batch_count=None):
    context_count = min(desired_context_count, dataset_context_count)
    batch_count = context_count//train_batch_size
    context_count = batch_count*train_batch_size
    if eval_batch_count is None:
        eval_batch_count = int((context_count*eval_percentage)//eval_batch_size)
    eval_context_count = eval_batch_count*eval_batch_size
    train_context_count = ((context_count-eval_context_count)//train_batch_size)*train_batch_size
    final_context_count = eval_context_count+train_context_count
    return train_context_count, eval_context_count

def prepare_dataset(config):
    all_contexts_dataset = load_from_disk('/home/cernypro/dev/source/ml4logs/data/interim/HDFS1_tokenized_chunked_size_10')
    train_context_count, eval_context_count = compute_train_eval_sizes(desired_context_count=config.how_many_contexts_to_use,
                                                                       dataset_context_count=len(all_contexts_dataset),
                                                                       train_batch_size=config.train_batch_size,
                                                                       eval_batch_size=config.eval_batch_size,
                                                                       eval_percentage=config.eval_ratio,
                                                                       eval_batch_count=config.eval_batch_count)
    train_eval_contexts = all_contexts_dataset.train_test_split(train_size=train_context_count, test_size=eval_context_count, shuffle=True, seed=config.seed)
    train_contexts = train_eval_contexts['train']
    eval_contexts = train_eval_contexts['test']

    rnd = np.random.default_rng(config.seed)

    NAME = f'Tr-{train_context_count}_Ev-{eval_context_count}_Epochs-{config.epochs}_Seed-{config.seed}'
    print(NAME)
    print(config)
    cur_dataset_path_dir = Path('/home/cernypro/dev/source/ml4logs/data/processed') / NAME


    train_dataset = train_contexts.map(prepare_ict,
                                       fn_kwargs={'epochs': config.epochs,
                                                  'rnd': rnd, 
                                                  'remove_target_prob': config.remove_target_percentage},
                                       batched=True, 
                                       batch_size=500, 
                                       remove_columns=train_contexts.column_names).shuffle(seed=config.seed)
    train_dataset.save_to_disk(cur_dataset_path_dir / 'train')
    print("Saved train")

    eval_dataset = eval_contexts.map(prepare_ict,
                                     fn_kwargs={'epochs': 1,
                                                'rnd': rnd, 
                                                'remove_target_prob': config.remove_target_percentage},
                                     batched=True, 
                                     batch_size=500, 
                                     remove_columns=eval_contexts.column_names)
    eval_dataset.save_to_disk(cur_dataset_path_dir / 'eval')
    print("Saved eval")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dataset preparation")
    parser.add_argument("--remove-target-percentage", default=0.9, type=float)
    parser.add_argument("--how-many-contexts-to-use", default=400000, type=int)
    parser.add_argument("--train-batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--eval-ratio", default=0.2, type=float)
    parser.add_argument("--eval-batch-size", default=64, type=int)
    parser.add_argument("--eval-batch-count", default=None, type=int, help="Overrides eval-ratio, will use eval-batch-count*eval-batch-size contexts for eval")
    parser.add_argument("--seed", default=43, type=int)

    config = parser.parse_args()
    prepare_dataset(config)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Seconds taken: {end - start}')
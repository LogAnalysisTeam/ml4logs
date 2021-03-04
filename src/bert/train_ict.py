import os
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from ict import DataCollatorForInverseClozeTask, OneTowerICT, TwoTowerICT

SEED = 42

def compute_dataset_sizes(desired_sentence_count, dataset_sentence_count, context_size, train_batch_size, eval_batch_size, eval_percentage=0.2):
    sentence_count = min(desired_sentence_count, dataset_sentence_count)
    context_count = sentence_count//context_size
    batch_count = context_count//train_batch_size
    context_count = batch_count*train_batch_size
    eval_batch_count = int((context_count*eval_percentage)//eval_batch_size)
    eval_context_count = eval_batch_count*eval_batch_size
    train_context_count = ((context_count-eval_context_count)//train_batch_size)*train_batch_size
    final_context_count = eval_context_count+train_context_count
    final_sentence_count = final_context_count*context_size
    return final_sentence_count, train_context_count, eval_context_count


def remove_timestamp(example):
    # need to find third occurence of a space and slice the string after it
    # using a very non robust silly solution
    s = example['text']
    example['text'] = s[s.find(' ', s.find(' ', s.find(' ')+1)+1)+1:]
    return example


def tokenize_no_special_tokens(examples, tokenizer):
    return {'tokens': tokenizer(examples['text'], add_special_tokens=False, truncation=True, return_attention_mask=False)['input_ids']}


def chunkify(examples):
    return {"chunk": [examples['tokens']]}


def run_experiment(config):
    os.environ["WANDB_PROJECT"] = "ICT"
    RUN_NAME = f'{"2T" if config.two_tower else "1T"} T-maxlen {config.target_max_seq_len} C-maxlen {config.context_max_seq_len} Tr-batch {config.train_batch_size} Ev-b {config.eval_batch_size} O-dim {config.output_encode_dim}'
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model, use_fast=True)
    data_collator = DataCollatorForInverseClozeTask(remove_target_from_context_probability=config.remove_target_percentage,
                                                    target_max_seq=config.target_max_seq_len,
                                                    context_max_seq=config.context_max_seq_len)

    hdfs1_dataset = load_dataset('text', data_files='../data/raw/HDFS1/HDFS.log', split='train')
    cleaned_dataset = hdfs1_dataset.map(remove_timestamp)
    sentence_count, train_contexts, eval_contexts = compute_dataset_sizes(desired_sentence_count=config.how_many_sentences_to_use,
                                                                          dataset_sentence_count=len(cleaned_dataset),
                                                                          context_size=config.context_sentence_count,
                                                                          train_batch_size=config.train_batch_size,
                                                                          eval_batch_size=config.eval_batch_size)

    subset_cleaned_dataset = cleaned_dataset.select(range(sentence_count))
    tokenized_dataset = subset_cleaned_dataset.map(tokenize_no_special_tokens, fn_kwargs={'tokenizer': tokenizer}, batched=True, batch_size=10000)
    chunked = tokenized_dataset.map(chunkify, batched=True, batch_size=config.context_sentence_count, drop_last_batch=True, remove_columns=tokenized_dataset.column_names)
    train_test_dataset = chunked.train_test_split(train_size=train_contexts, test_size=eval_contexts, shuffle=True, seed=SEED)

    model = TwoTowerICT(config.bert_model, output_encode_dimension=config.output_encode_dim) if config.two_tower else OneTowerICT(config.bert_model, output_encode_dimension=output_encode_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    training_args = TrainingArguments(output_dir=f"../models/{RUN_NAME.replace(' ', '_')}",
                                      num_train_epochs=2,
                                      per_device_eval_batch_size=config.eval_batch_size, 
                                      per_device_train_batch_size=config.train_batch_size,
                                      warmup_steps=100,                # number of warmup steps for learning rate scheduler
                                      weight_decay=0.01,               # strength of weight decay
                                      logging_dir='../logs',            # directory for storing logs
                                      logging_steps=10,
                                      logging_first_step=True,
                                      eval_steps=50,
                                      evaluation_strategy='steps',
                                      prediction_loss_only=True,
                                      save_steps=100,
                                      save_total_limit=15,
                                      label_names=['target', 'context'],
                                      seed=SEED,
                                      run_name=RUN_NAME,
                                      remove_unused_columns=False)

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_test_dataset['train'],
                      eval_dataset=train_test_dataset['test']
                      )

    trainer.train()
    trainer.save_model()
    

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Runner for ICT experiments")
    parser.add_argument('--two-tower', default=False, action='store_true', help="Use TwoTowerICT")
    parser.add_argument('--bert-model', default="distilbert-base-cased", type=str, help="Pretrained Transformer for the encoder towers.")
    # parser.add_argument("--bert-model-target", 
    #                     default="distilbert-base-cased", 
    #                     type=str, help="Pretrained Transformer for target sentence tower.")
    # parser.add_argument("--bert-model-context", 
    #                     default="distilbert-base-cased", 
    #                     type=str, help="Pretrained Transformer for context sentence tower. (only when using Two Tower model)")
    parser.add_argument("--remove-target-percentage", default=0.9, type=float)
    parser.add_argument("--context-sentence-count", default=10, type=int)
    parser.add_argument("--how-many-sentences-to-use", default=1000000, type=int)
    parser.add_argument("--train-batch-size", default=64, type=int)
    parser.add_argument("--eval-batch-size", default=64, type=int)
    parser.add_argument("--target-max-seq-len", default=512, type=int)
    parser.add_argument("--context-max-seq-len", default=512, type=int)
    parser.add_argument("--output-encode-dim", default=512, type=int, help="Output dimension for the encoder towers")

    config = parser.parse_args()
    run_experiment(config)


if __name__ == '__main__':
    main()
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import numpy as np
import torch
from torch.utils.data import DataLoader

SEED = 42
RUN_NAME = "DistilBERT HDFS MLM pretraining 1"
pretrained_model_name = "distilbert-base-cased"


def remove_timestamp(example):
    # need to find third occurence of a space and slice the string after it
    # using a very non robust silly solution
    s = example['text']
    example['text'] = s[s.find(' ', s.find(' ', s.find(' ')+1)+1)+1:]
    return example

def tokenize_dontpad_dataset(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, return_special_tokens_mask=True)


if __name__ == '__main__':
    torch.manual_seed(SEED)
    tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    hdfs1_dataset = load_dataset('text', data_files='../../data/raw/HDFS1/HDFS.log', split='train')
    cleaned_dataset = hdfs1_dataset.map(remove_timestamp)
    
    tokenized_unpadded_dataset = cleaned_dataset.map(tokenize_dontpad_dataset, fn_kwargs={'tokenizer': tokenizer}, batched=True, batch_size=1000, remove_columns=['text'])
    
    train_test_dataset = tokenized_unpadded_dataset.train_test_split(test_size=50000, shuffle=True, seed=SEED)
    
    model = DistilBertForMaskedLM.from_pretrained(pretrained_model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    training_args = TrainingArguments(output_dir=f"../../models/{RUN_NAME.replace(' ', '_')}",
                                      num_train_epochs=5,
                                      per_device_eval_batch_size=256, 
                                      per_device_train_batch_size=128,
                                      warmup_steps=500,                # number of warmup steps for learning rate scheduler
                                      weight_decay=0.01,               # strength of weight decay
                                      logging_dir='./logs',            # directory for storing logs
                                      logging_steps=50,
                                      logging_first_step=True,
                                      eval_steps=500,
                                      evaluation_strategy='steps',
                                      prediction_loss_only=True,
                                      save_steps=2000,
                                      save_total_limit=15,
                                      seed=SEED,
                                      run_name=RUN_NAME)
    
    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_test_dataset['train'],
                      eval_dataset=train_test_dataset['test']
                      )
    
    trainer.train()
    trainer.save_model()
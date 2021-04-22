from dataclasses import dataclass
import torch
from typing import List, Union, Dict
import numpy as np
from transformers import AutoModel

from pathlib import Path


def _make_mask(padded_batch, pad_token=0):
    return (padded_batch != pad_token).to(torch.uint8)


def _pad_truncate_add_special_tokens(batch: List[List[int]], max_len, pad_token=0, start_token=101, sep_token=102):
    sequence_lengths = torch.tensor([min(max_len-2, len(seq)) for seq in batch], dtype=torch.int64)
    batch_max_len = sequence_lengths.max()
    padded_batch = torch.full(size=(len(batch), batch_max_len+2), fill_value=pad_token, dtype=torch.int64)
    padded_batch[:, 0] = start_token
    for seq_idx, seq in enumerate(batch):
        padded_batch[seq_idx, 1:sequence_lengths[seq_idx]+1] = torch.tensor(seq[:sequence_lengths[seq_idx]], dtype=torch.int64)
        padded_batch[seq_idx, sequence_lengths[seq_idx]+1] = sep_token
    mask = _make_mask(padded_batch)
    return padded_batch, mask


@dataclass
class DataCollatorForPreprocessedICT:
    target_max_seq:int = 512
    context_max_seq:int = 512
    start_token:int = 101 # [CLS]
    sep_token:int = 102 # [SEP]
    pad_token:int = 0
        
    def _pad_truncate_add_special_tokens(self, batch: List[List[int]], max_len):
        return _pad_truncate_add_special_tokens(batch, max_len=max_len, pad_token=self.pad_token, start_token=self.start_token, sep_token=self.sep_token)
             
    def __call__(self, contexts: List[Dict[str, List[int]]]):
        if isinstance(contexts[0], dict):
            target_sentences = [context_dict['target'] for context_dict in contexts]
            flattened_contexts = [context_dict['flat_context'] for context_dict in contexts]
        correct_class = torch.arange(len(target_sentences), dtype=torch.int64)
        padded_target_batch, padded_target_mask = self._pad_truncate_add_special_tokens(target_sentences, self.target_max_seq)
        padded_context_batch, padded_context_mask = self._pad_truncate_add_special_tokens(flattened_contexts, self.context_max_seq)
        return {'target': padded_target_batch,
                'target_mask': padded_target_mask,
                'context': padded_context_batch,
                'context_mask': padded_context_mask,
                'correct_class': correct_class}

@dataclass
class DataCollatorForInverseClozeTask:
    """
    Data Collator to be used with datasets containing contexts (List of senteces, where sentence is a list of tokens, e.g a context is a List of Lists of ints)
    It randomly selects a sentence from each context to serve as a target, and flattens the contexts (with the target sentence randomly removed) into a single sentence to serve as a flat context
    If using huggingface datasets, expects a column named 'chunk'
    NOTE: This collator is quite slow, if the dataset is really large, consider pre-collating it in advance and saving it, so that the collator can then be as simple as possible
        (example of how slow it is, when resuming almost finished training on a dataset of 800k contexts (each containing 10 lines, eg 8M lines in total)
        using batches of size 64, resuming training took over 5 hours on 4 cores, because all the intermediate batches were created by the dataloader and collated by this collater
        until the Trainer (huggingface) reached the saved step in the checkpoint)
    """
    remove_target_from_context_probability: float = 0.9
    target_max_seq:int = 512
    context_max_seq:int = 512
    start_token:int = 101 # [CLS]
    sep_token:int = 102 # [SEP]
    pad_token:int = 0
        
    def _pad_truncate_add_special_tokens(self, batch: List[List[int]], max_len): 
        return _pad_truncate_add_special_tokens(batch, max_len=max_len, pad_token=self.pad_token, start_token=self.start_token, sep_token=self.sep_token)
    
    def _create_target_and_flat_contexts_from_contexts(self, contexts: List[List[List[int]]]):
        # TODO: add sep_token between each sentence when flattening context?
        target_sentence_idxs = [torch.randint(low=0, high=len(context), size=(1,)).item() for context in contexts]
        remove_target = [torch.rand(size=(1,)).item() < self.remove_target_from_context_probability for _ in target_sentence_idxs]
        target_sentences = [context[i] for (i, context) in zip(target_sentence_idxs, contexts)]
        processed_contexts = [context[:target_idx] + context[target_idx + remove:] for (target_idx, remove, context) in zip(target_sentence_idxs, remove_target, contexts)]
        flattened_contexts = [[token for sentence in context for token in sentence] for context in processed_contexts]
        return target_sentences, flattened_contexts
             
    def __call__(self, contexts: List[Union[List[List[int]], Dict[str, List[List[int]]]]]):
        if isinstance(contexts[0], dict):
            contexts = [context_dict['chunk'] for context_dict in contexts]
        target_sentences, flattened_contexts = self._create_target_and_flat_contexts_from_contexts(contexts)
        correct_class = torch.arange(len(target_sentences), dtype=torch.int64)
        padded_target_batch, padded_target_mask = self._pad_truncate_add_special_tokens(target_sentences, self.target_max_seq)
        padded_context_batch, padded_context_mask = self._pad_truncate_add_special_tokens(flattened_contexts, self.context_max_seq)
        return {'target': padded_target_batch,
                'target_mask': padded_target_mask,
                'context': padded_context_batch,
                'context_mask': padded_context_mask,
                'correct_class': correct_class}


class ClsEncoderTower(torch.nn.Module):
    """
    Simple model on top of a BERT like model.
    It's a linear layer on the [CLS] token of each sentence from BERT.
    """
    def __init__(self, pretrained_model_name_or_path, output_encode_dimension=512):
        super(ClsEncoderTower, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.linear = torch.nn.Linear(self.bert.config.dim, output_encode_dimension) # self.bert.config.dim most likely 768
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_embedding = bert_output[0][:, 0]
        cls_encoding = self.linear(cls_token_embedding)
        return cls_encoding


class OneTowerICT(torch.nn.Module):
    """
    Network for the inverse close task, uses one BERT tower for creating encodings of target and context sentences (query and document as per nomenclature of original paper)
    Uses cross entropy loss
    """
    def __init__(self, tower_class, pretrained_model_name_or_path, output_encode_dimension=100):
        super(OneTowerICT, self).__init__()
        self.tower = tower_class.from_pretrained(pretrained_model_name_or_path,
                                                               task_specific_params={'cls_embedding_dimension': output_encode_dimension})
        self.loss_fn = torch.nn.CrossEntropyLoss()
    def forward(self, target, target_mask, context, context_mask, correct_class):
        target_cls_encode = self.tower(input_ids=target, attention_mask=target_mask).embedding
        context_cls_encode = self.tower(input_ids=context, attention_mask=context_mask).embedding
        
        logits = torch.matmul(target_cls_encode, context_cls_encode.transpose(-2, -1))
        loss = self.loss_fn(logits, correct_class)
        return loss, target_cls_encode, context_cls_encode

    def save_encoder(self, name: str, basedir: Path):
        encoder_name = f'LogEncoder_from_{name.replace(" ", "_")}'
        encoder_path = basedir / encoder_name
        self.tower.save_pretrained(encoder_path)



class TwoTowerICT(torch.nn.Module):
    def __init__(self, tower_class, target_tower_pretrained_model_name_or_path, context_tower_pretrained_model_name_or_path=None, output_encode_dimension=100):
        super(TwoTowerICT, self).__init__()
        assert target_tower_pretrained_model_name_or_path is not None, "Target tower pretrained model must me specified!"
        if context_tower_pretrained_model_name_or_path is None:
            context_tower_pretrained_model_name_or_path = target_tower_pretrained_model_name_or_path
        self.target_encoder = tower_class.from_pretrained(target_tower_pretrained_model_name_or_path, task_specific_params={'cls_embedding_dimension': output_encode_dimension})
        self.context_encoder = tower_class.from_pretrained(context_tower_pretrained_model_name_or_path, task_specific_params={'cls_embedding_dimension': output_encode_dimension})
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, target, target_mask, context, context_mask, correct_class):
        target_cls_encode = self.target_encoder(input_ids=target, attention_mask=target_mask).embedding
        context_cls_encode = self.context_encoder(input_ids=context, attention_mask=context_mask).embedding
        
        logits = torch.matmul(target_cls_encode, context_cls_encode.transpose(-2, -1))
        loss = self.loss_fn(logits, correct_class)
        return loss, target_cls_encode, context_cls_encode

    def save_encoder(self, name: str, basedir: Path):
        encoder_name = f'LogEncoder_from_{name.replace(" ", "_")}'
        encoder_path = basedir / encoder_name
        self.target_encoder.save_pretrained(encoder_path)
        context_encoder_name = f'ContextEncoder_from_{name.replace(" ", "_")}'
        context_encoder_path = basedir / context_encoder_name
        self.context_encoder.save_pretrained(context_encoder_path)
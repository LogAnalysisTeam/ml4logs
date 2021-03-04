from dataclasses import dataclass
import torch
from typing import List, Union, Dict
import numpy as np
import torch
from transformers import AutoModel

@dataclass
class DataCollatorForInverseClozeTask:
    """
    Data Collator to be used with datasets containing contexts (List of senteces, where sentence is a list of tokens, e.g a context is a List of Lists of ints)
    It randomly selects a sentence from each context to serve as a target, and flattens the contexts (with the target sentence randomly removed) into a single sentence to serve as a flat context
    If using huggingface datasets, expects a column named 'chunk'
    """
    remove_target_from_context_probability: float = 0.9
    target_max_seq:int = 512
    context_max_seq:int = 512
    start_token:int = 101 # [CLS]
    sep_token:int = 102 # [SEP]
    pad_token:int = 0
        
    def _make_mask(self, padded_batch):
        return (padded_batch != self.pad_token).astype(np.uint8)
        
    def _pad_truncate_add_special_tokens(self, batch: List[List[int]], max_len): 
        sequence_lengths = np.array([min(max_len-2, len(seq)) for seq in batch])
        batch_max_len = sequence_lengths.max()
        padded_batch = np.full(shape=(len(batch), batch_max_len+2), fill_value=self.pad_token, dtype=np.int64)
        padded_batch[:, 0] = self.start_token
        for seq_idx, seq in enumerate(batch):
            padded_batch[seq_idx, 1:sequence_lengths[seq_idx]+1] = seq[:sequence_lengths[seq_idx]]
            padded_batch[seq_idx, sequence_lengths[seq_idx]+1] = self.sep_token
        mask = self._make_mask(padded_batch)
        return torch.from_numpy(padded_batch), torch.from_numpy(mask)
    
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
        correct_class = torch.tensor(list(range(len(target_sentences))), dtype=torch.int64)
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
    def __init__(self, pretrained_model_name_or_path, output_encode_dimension=512):
        super(OneTowerICT, self).__init__()
        self.tower = ClsEncoderTower(pretrained_model_name_or_path, output_encode_dimension)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    def forward(self, target, target_mask, context, context_mask, correct_class):
        target_cls_encode = self.tower(input_ids=target, attention_mask=target_mask)
        context_cls_encode = self.tower(input_ids=context, attention_mask=context_mask)
        
        logits = torch.matmul(target_cls_encode, context_cls_encode.transpose(-2, -1))
        loss = self.loss_fn(logits, correct_class)
        return loss, target_cls_encode, context_cls_encode


class TwoTowerICT(torch.nn.Module):
    def __init__(self, target_tower_pretrained_model_name_or_path, context_tower_pretrained_model_name_or_path=None, output_encode_dimension=512):
        super(TwoTowerICT, self).__init__()
        assert target_tower_pretrained_model_name_or_path is not None, "Target tower pretrained model must me specified!"
        if context_tower_pretrained_model_name_or_path is None:
            context_tower_pretrained_model_name_or_path = target_tower_pretrained_model_name_or_path
        self.target_encoder = ClsEncoderTower(target_tower_pretrained_model_name_or_path, output_encode_dimension)
        self.context_encoder = ClsEncoderTower(context_tower_pretrained_model_name_or_path, output_encode_dimension)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, target, target_mask, context, context_mask, correct_class):
        target_cls_encode = self.target_encoder(input_ids=target, attention_mask=target_mask)
        context_cls_encode = self.context_encoder(input_ids=context, attention_mask=context_mask)
        
        logits = torch.matmul(target_cls_encode, context_cls_encode.transpose(-2, -1))
        loss = self.loss_fn(logits, correct_class)
        return loss, target_cls_encode, context_cls_encode
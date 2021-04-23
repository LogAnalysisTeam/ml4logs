from transformers.file_utils import ModelOutput
from transformers import DistilBertModel, DistilBertPreTrainedModel, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
import torch
from dataclasses import dataclass


@dataclass
class EmbeddingOutput(ModelOutput):
    """
    ModelOutput class inspired per Huggingface Transformers library conventions, may be replaced by a suitable alternative class from the library if any exists.
    """
    embedding: torch.FloatTensor = None


class DistilBertForClsEmbedding(DistilBertPreTrainedModel):
    """
    DistilBertModel with a linear layer applied to [CLS] token.
    Initialize using .from_pretrained(path_or_model_name) method
    use task_specific_params={'cls_embedding_dimension': *YOUR EMBEDDING DIMENSION HERE*} to set embedding dimension
    """
    def __init__(self, config):
        super().__init__(config)
        if config.task_specific_params is None:
            config.task_specific_params = dict()

        self.distilbert = DistilBertModel(config)
        self.cls_projector = torch.nn.Linear(config.dim, config.task_specific_params.setdefault('cls_embedding_dimension', 100))

        self.init_weights()
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_embedding = bert_output.last_hidden_state[:, 0]
        cls_encoding = self.cls_projector(cls_token_embedding)
        return EmbeddingOutput(embedding=cls_encoding)


class DistilBertForMeanEmbedding(DistilBertPreTrainedModel):
    """
    DistilBertModel with a linear layer applied to mean of non-special tokens (Tokens except first and last in positive examples in attention mask).
    Initialize using .from_pretrained(path_or_model_name) method
    use task_specific_params={'cls_embedding_dimension': *YOUR EMBEDDING DIMENSION HERE*} to set embedding dimension
    """
    def __init__(self, config):
        super().__init__(config)
        if config.task_specific_params is None:
            config.task_specific_params = dict()

        self.distilbert = DistilBertModel(config)
        self.mean_projector = torch.nn.Linear(config.dim, config.task_specific_params.setdefault('cls_embedding_dimension', 100))

        self.init_weights()
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        non_special_tokens_mask = attention_mask.clone()
        non_special_tokens_mask[:, 0] = 0
        non_special_tokens_mask = torch.roll(non_special_tokens_mask, -1, dims=1)
        non_special_tokens_mask[:, 0] = 0
        
        output_masked = bert_output.last_hidden_state * non_special_tokens_mask.unsqueeze(-1).float()
        mean_tokens_embedding = output_masked.sum(dim=1)/non_special_tokens_mask.sum(dim=1).unsqueeze(-1).float()
        mean_encoding = self.mean_projector(mean_tokens_embedding)
        return EmbeddingOutput(embedding=mean_encoding)


class RobertaForClsEmbedding(RobertaPreTrainedModel):
    """
    RobertaModel with a linear layer applied to [CLS] token.
    Initialize using .from_pretrained(path_or_model_name) method
    use task_specific_params={'cls_embedding_dimension': *YOUR EMBEDDING DIMENSION HERE*} to set embedding dimension
    """
    def __init__(self, config):
        super().__init__(config)
        if config.task_specific_params is None:
            config.task_specific_params = dict()

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.cls_projector = torch.nn.Linear(config.hidden_size, config.task_specific_params.setdefault('cls_embedding_dimension', 100))

        self.init_weights()
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_embedding = bert_output.last_hidden_state[:, 0]
        cls_encoding = self.cls_projector(cls_token_embedding)
        return EmbeddingOutput(embedding=cls_encoding)


class RobertaForMeanEmbedding(RobertaPreTrainedModel):
    """
    RobertaModel with a linear layer applied to mean of non-special tokens (Tokens except first and last in positive examples in attention mask).
    Initialize using .from_pretrained(path_or_model_name) method
    use task_specific_params={'cls_embedding_dimension': *YOUR EMBEDDING DIMENSION HERE*} to set embedding dimension
    """
    def __init__(self, config):
        super().__init__(config)
        if config.task_specific_params is None:
            config.task_specific_params = dict()

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.mean_projector = torch.nn.Linear(config.hidden_size, config.task_specific_params.setdefault('cls_embedding_dimension', 100))

        self.init_weights()
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        non_special_tokens_mask = attention_mask.clone()
        non_special_tokens_mask[:, 0] = 0
        non_special_tokens_mask = torch.roll(non_special_tokens_mask, -1, dims=1)
        non_special_tokens_mask[:, 0] = 0
        
        output_masked = bert_output.last_hidden_state * non_special_tokens_mask.unsqueeze(-1).float()
        mean_tokens_embedding = output_masked.sum(dim=1)/non_special_tokens_mask.sum(dim=1).unsqueeze(-1).float()
        mean_encoding = self.mean_projector(mean_tokens_embedding)
        return EmbeddingOutput(embedding=mean_encoding)

# possible method for doing masked mean along y -> output shape (x, z)
# data - shape (x, y, z)
# mask - shape (x, y)
# data_masked = data * mask.unsqueeze(-1).float() # this zeroes out the vectors along y to not use
# data_masked.sum(dim=1)/mask.sum(dim=1).unsqueeze(-1)  # sums up y, and then divides each vector by amount of unmasked elements

# possible method to shorten mask on both sides, with each row of type [1 1 1 1 0 0], e.g. full of 1s up to a point and then 0s, with the goal being transforming the first and last 1s into zeros, e.g. for the example row have it become [0 1 1 0 0 0]
# mask - shape (x, y)
# mask[:, 0] = 0  # roll wraps around, have to zero-out first column, because it will become the last
# mask = torch.roll(mask, -1, dims=1)
# mask[:, 0] = 0 # zero-out the first column for real


### MUST BE LAST IN FILE, NOT REALLY CLEAN, BUT DONE FOR IMPORTING
KNOWN_ENCODER_CLASSES = {
    'DistilbertCls': DistilBertForClsEmbedding,
    'RobertaCls': RobertaForClsEmbedding,
    'DistilbertMean': DistilBertForMeanEmbedding,
    'RobertaMean': RobertaForMeanEmbedding
}
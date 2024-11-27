import torch
import torch.nn as nn
import torch.optim as optim 

from bert import BERTEncoder
from embedding import EmbeddingLayer

class BERTModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_position_embeddings, type_vocab_size):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, hidden_size, max_position_embeddings, type_vocab_size)
        self.encoder = BERTEncoder(num_layers, hidden_size, num_heads, intermediate_size)

    def forward(self, input_ids, token_type_ids, mask=None):
        embeddings = self.embedding(input_ids, token_type_ids)
        return self.encoder(embeddings, mask)

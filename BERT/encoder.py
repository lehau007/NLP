import torch
import torch.nn as nn
import torch.optim as optim

from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForwardNetwork

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.ffn = FeedForwardNetwork(hidden_size, intermediate_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, mask=None):
        # Multi-head attention sublayer
        attention_output = self.attention(hidden_states, mask)
        hidden_states = self.layer_norm1(hidden_states + self.dropout(attention_output))

        # Feedforward network sublayer
        ffn_output = self.ffn(hidden_states)
        return self.layer_norm2(hidden_states + self.dropout(ffn_output))

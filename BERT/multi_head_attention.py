import torch
import torch.nn as nn
import torch.optim as optim

from scale_dot import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, mask=None):
        batch_size, seq_len, hidden_size = hidden_states.size()
        # Project input to Q, K, V
        query = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_output = ScaledDotProductAttention(query, key, value, mask)

        # Concatenate heads and pass through the output layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.out(attention_output)


import torch
import torch.nn as nn
import torch.optim as optim

from encoder import TransformerBlock

class BERTEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states, mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        return hidden_states

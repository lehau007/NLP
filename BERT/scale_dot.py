import torch
import torch.nn as nn
import math

def ScaledDotProductAttention(query, key, value, mask=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        # Ensure mask shape matches scores
        mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        mask = mask.expand(-1, scores.size(1), scores.size(2), scores.size(3))
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
    attention_output = torch.matmul(attention_weights, value)
    return attention_output
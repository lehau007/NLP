from transformers import BertTokenizer
from Bert_model import BERTModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import random
import torch

def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    
    # Use tokenizer to get special tokens mask
    special_tokens_mask = tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # Replace 80% with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # Replace 10% with random tokens
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # 10% remain unchanged
    return input_ids, labels


def prepare_nsp_data(sentences, tokenizer):
    input_ids = []
    token_type_ids = []
    labels = []

    for sentence_a, sentence_b, is_next in sentences:
        tokens_a = tokenizer.tokenize(sentence_a)
        tokens_b = tokenizer.tokenize(sentence_b)

        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

        input_ids.append(tokenizer.convert_tokens_to_ids(tokens))
        token_type_ids.append(segment_ids)
        labels.append(1 if is_next else 0)

    return input_ids, token_type_ids, labels

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class BERTDataset(Dataset):
    def __init__(self, inputs, token_types, labels, tokenizer, max_len):
        self.inputs = inputs
        self.token_types = token_types
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]
        token_type_ids = self.token_types[idx]
        label = self.labels[idx]

        padding_length = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        token_type_ids += [0] * padding_length

        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(label)

# Losses for MLM and NSP
mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
nsp_loss_fn = nn.CrossEntropyLoss()

# Training loop
def train_bert(model, dataloader, optimizer, tokenizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, token_type_ids, labels in dataloader:
            input_ids, token_type_ids, labels = input_ids.to(device), token_type_ids.to(device), labels.to(device)

            # Forward pass
            outputs = model(input_ids, token_type_ids)

            # MLM loss
            mlm_input_ids, mlm_labels = mask_tokens(input_ids, tokenizer)
            mlm_outputs = outputs[0]  # MLM logits
            mlm_loss = mlm_loss_fn(mlm_outputs.view(-1, tokenizer.vocab_size), mlm_labels.view(-1))

            # NSP loss
            nsp_logits = outputs[1]  # NSP logits
            nsp_loss = nsp_loss_fn(nsp_logits, labels)

            # Total loss
            loss = mlm_loss + nsp_loss
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

from torch.optim import AdamW
from transformers import get_scheduler

model = BERTModel(50000, 512, 13, 8, 512, 512, 50000)
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=1000, num_training_steps=3)


torch.save(model.state_dict(), "bert_from_scratch.pth")

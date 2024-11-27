import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from Bert_model import BERTModel
from model_train import mask_tokens

VOCAB_SIZE = 30522  # Standard BERT vocab size
HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
INTERMEDIATE_SIZE = 3072
MAX_POSITION_EMBEDDINGS = 512
TYPE_VOCAB_SIZE = 2
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4

# Dataset
class BERTDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_len):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence_a, sentence_b, is_next = self.sentences[idx]

        # Tokenize and encode
        inputs = self.tokenizer.encode_plus(
            sentence_a,
            sentence_b,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].squeeze(0)
        token_type_ids = inputs["token_type_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Mask tokens
        input_ids, mlm_labels = mask_tokens(input_ids, self.tokenizer)

        return input_ids, token_type_ids, attention_mask, mlm_labels, torch.tensor(is_next)

# Bert
class BERTForPreTraining(BERTModel):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_position_embeddings, type_vocab_size):
        super().__init__(vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_position_embeddings, type_vocab_size)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.nsp_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, token_type_ids, mask=None):
        embeddings = self.embedding(input_ids, token_type_ids)
        encoded_output = self.encoder(embeddings, mask)

        # [CLS] token's output for NSP
        cls_output = encoded_output[:, 0, :]
        nsp_logits = self.nsp_head(cls_output)

        # MLM logits
        mlm_logits = self.mlm_head(encoded_output)

        return mlm_logits, nsp_logits

# Training Func 
def train_bert(model, dataloader, optimizer, tokenizer, device):
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for input_ids, token_type_ids, attention_mask, mlm_labels, nsp_labels in dataloader:
            input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
            mlm_labels, nsp_labels = mlm_labels.to(device), nsp_labels.to(device)

            # Forward pass
            mlm_logits, nsp_logits = model(input_ids, token_type_ids, attention_mask)

            # Compute losses
            mlm_loss = mlm_loss_fn(mlm_logits.view(-1, tokenizer.vocab_size), mlm_labels.view(-1))
            nsp_loss = nsp_loss_fn(nsp_logits, nsp_labels)
            loss = mlm_loss + nsp_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Main Function
def main():
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Example dataset:
    sentences = [
        ("This is the first sentence.", "This is the second sentence.", 1),
        ("This is a sentence.", "A completely unrelated sentence.", 0),
    ]

    # From data create dataset and dataloader for pytorch training
    dataset = BERTDataset(sentences, tokenizer, MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTForPreTraining(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        type_vocab_size=TYPE_VOCAB_SIZE,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training 
    train_bert(model, dataloader, optimizer, tokenizer, device)

    # Save model
    torch.save(model.state_dict(), "bert_from_scratch.pth")

if __name__ == "__main__":
    main()

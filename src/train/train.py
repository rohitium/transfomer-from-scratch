# src/train/train.py
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from src.models.transformer import Transformer

# Example dataset class
class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_pad_idx=0, tgt_pad_idx=0):
        # Load your BPE-encoded lines
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_data = [line.strip().split() for line in f]
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_data = [line.strip().split() for line in f]
        assert len(self.src_data) == len(self.tgt_data)
        
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

    # A custom collate_fn might be needed to batch variable-length sequences
    @staticmethod
    def collate_fn(batch):
        # batch = [([src_tokens], [tgt_tokens]), ...]
        src_seqs, tgt_seqs = zip(*batch)

        # Get max lengths
        src_max_len = max(len(seq) for seq in src_seqs)
        tgt_max_len = max(len(seq) for seq in tgt_seqs)

        # Initialize padded tensors
        src_padded = torch.full((len(batch), src_max_len), 0, dtype=torch.long)  # 0 is pad_idx
        tgt_padded = torch.full((len(batch), tgt_max_len), 0, dtype=torch.long)  # 0 is pad_idx

        # Fill padded tensors with token ids
        for i, (src_seq, tgt_seq) in enumerate(zip(src_seqs, tgt_seqs)):
            # Convert tokens to integers (assuming tokens are already integers from BPE encoding)
            src_ids = [int(token) for token in src_seq]
            tgt_ids = [int(token) for token in tgt_seq]
            
            # Copy to padded tensors
            src_padded[i, :len(src_ids)] = torch.tensor(src_ids)
            tgt_padded[i, :len(tgt_ids)] = torch.tensor(tgt_ids)

        return src_padded, tgt_padded

def generate_subsequent_mask(size):
    """
    Creates a lower-triangular matrix for causal masking in the decoder.
    (size x size), used for preventing the decoder from attending to future tokens.
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        
        # We'll shift the target for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        # Generate causal mask for decoder
        seq_len = tgt_input.size(1)
        tgt_mask = generate_subsequent_mask(seq_len).to(device)

        optimizer.zero_grad()
        outputs = model(src, tgt_input, tgt_mask=tgt_mask)
        
        # Flatten outputs for cross-entropy
        outputs = outputs.reshape(-1, outputs.size(-1))
        tgt_labels = tgt_labels.reshape(-1)
        
        loss = criterion(outputs, tgt_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    # Hyperparams
    SRC_VOCAB_SIZE = 32000
    TGT_VOCAB_SIZE = 32000
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_LAYERS = 6
    DROPOUT = 0.1
    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 1e-4

    # Create Dataset and DataLoader
    train_dataset = TranslationDataset("data/processed/train.en.bpe", "data/processed/train.de.bpe")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=TranslationDataset.collate_fn)

    # Initialize model, optimizer, loss
    model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignoring <PAD> index as example

    # Training loop
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss:.4f}")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/transformer_base.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()

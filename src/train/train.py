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
    def __init__(self, src_file, tgt_file, src_vocab=None, tgt_vocab=None, 
                 src_pad_idx=0, tgt_pad_idx=0, max_len=128):  # Reduced max_len
        super().__init__()
        # Load and filter data
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_data = []
            for line in f:
                tokens = line.strip().split()
                if len(tokens) <= max_len:  # Only keep sequences within max_len
                    self.src_data.append(tokens)
        
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_data = []
            for line in f:
                tokens = line.strip().split()
                if len(tokens) <= max_len:  # Only keep sequences within max_len
                    self.tgt_data.append(tokens)
        
        # Ensure src and tgt have same number of filtered sequences
        min_len = min(len(self.src_data), len(self.tgt_data))
        self.src_data = self.src_data[:min_len]
        self.tgt_data = self.tgt_data[:min_len]
        
        self.src_vocab = src_vocab or self._build_vocab(self.src_data)
        self.tgt_vocab = tgt_vocab or self._build_vocab(self.tgt_data)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.max_len = max_len

        total_sequences = len(self.src_data)
        print(f"Total sequences after filtering (max_len={max_len}): {total_sequences}")
        print("src_data length:", len(self.src_data))
        print("tgt_data length:", len(self.tgt_data))

        print(f"Filtered out {len(self.src_data) + len(self.tgt_data) - total_sequences} sequences")

    def _build_vocab(self, data):
        # Create a simple vocabulary from the data
        vocab = {'<pad>': 0, '<unk>': 1}
        for sentence in data:
            for token in sentence:
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        # Convert tokens to indices using vocabulary, handling unknown tokens
        src_indices = [self.src_vocab[token] for token in self.src_data[idx]]
        tgt_indices = [self.tgt_vocab[token] for token in self.tgt_data[idx]]
        return src_indices, tgt_indices

    @staticmethod
    def collate_fn(batch):
        src_seqs, tgt_seqs = zip(*batch)
        
        # Get max lengths
        src_max_len = max(len(seq) for seq in src_seqs)
        tgt_max_len = max(len(seq) for seq in tgt_seqs)

        # Initialize padded tensors
        src_padded = torch.full((len(batch), src_max_len), 0, dtype=torch.long)
        tgt_padded = torch.full((len(batch), tgt_max_len), 0, dtype=torch.long)

        for i, (src_seq, tgt_seq) in enumerate(zip(src_seqs, tgt_seqs)):
            # No need for conversion since sequences are already indices
            src_padded[i, :len(src_seq)] = torch.tensor(src_seq)
            tgt_padded[i, :len(tgt_seq)] = torch.tensor(tgt_seq)

        return src_padded, tgt_padded

def create_pad_mask(seq, pad_idx=0):
    """
    Creates a 2D pad mask: (B, T), where 1 = valid token, 0 = pad.
    We'll expand it later in expand_pad_mask().
    """
    return (seq != pad_idx).long()

def expand_pad_mask(mask_2d):
    """
    Convert a (B, T) 2D mask into (B, 1, T, T) for self-attention.
    1 = allowed token, 0 = masked/padded.
    """
    B, T = mask_2d.shape
    # Step 1: (B, T) -> (B, 1, T)
    mask_3d = mask_2d.unsqueeze(1)
    # Step 2: (B, 1, T) -> (B, 1, 1, T)
    mask_4d = mask_3d.unsqueeze(2)
    # Step 3: expand last dim from 1 -> T
    mask_4d = mask_4d.expand(B, 1, T, T)
    return mask_4d

def generate_subsequent_mask(size):
    """Generate causal mask for decoder's self attention"""
    mask = (1 - torch.triu(torch.ones(size, size), diagonal=1))
    return mask.unsqueeze(0)  # (1, size, size)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        
        # We'll shift the target for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        # 1) Build source pad mask => 2D (B, T_src)
        src_pad_2d = create_pad_mask(src, pad_idx=0).to(device)   # shape (B, T_src)
        # Expand to 4D => (B, 1, T_src, T_src) for self-attention
        src_mask_4d = expand_pad_mask(src_pad_2d)                 # shape (B, 1, T_src, T_src)

        # 2) Build target pad mask => 2D (B, T_tgt)
        tgt_pad_2d = create_pad_mask(tgt_input, pad_idx=0).to(device)  # shape (B, T_tgt)
        # Expand to 4D => (B, 1, T_tgt, T_tgt)
        tgt_pad_4d = expand_pad_mask(tgt_pad_2d)

        # 3) Generate causal lower-triangular mask => (1, T_tgt, T_tgt)
        causal_3d = generate_subsequent_mask(tgt_input.size(1)).to(device)  # shape (1, T_tgt, T_tgt)

        # Expand causal from (1, T, T) to (B, 1, T, T)
        B = tgt_input.size(0)
        causal_4d = causal_3d.expand(B, 1, -1, -1)  # shape (B, 1, T_tgt, T_tgt)

        # 4) Combine them: final_tgt_mask is the elementwise product of (pad mask & causal mask)
        #   Because we stored them as 1=allowed, 0=blocked, use * to combine
        final_tgt_mask = (tgt_pad_4d * causal_4d).to(device)  # shape (B, 1, T_tgt, T_tgt)

        # Now we can pass src_mask_4d and final_tgt_mask to the model
        optimizer.zero_grad()
        outputs = model(
            src,
            tgt_input,
            src_mask=src_mask_4d,
            tgt_mask=final_tgt_mask
        )
        
        # Flatten outputs for cross-entropy
        outputs = outputs.reshape(-1, outputs.size(-1))
        tgt_labels = tgt_labels.reshape(-1)
        
        loss = criterion(outputs, tgt_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

class Vocabulary:
    def __init__(self, pad_token='<pad>', unk_token='<unk>'):
        self.token2idx = {pad_token: 0, unk_token: 1}
        self.idx2token = {0: pad_token, 1: unk_token}
    
    def add_token(self, token):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
    
    def __getitem__(self, token):
        return self.token2idx.get(token, self.token2idx['<unk>'])
    
    def __len__(self):
        return len(self.token2idx)

def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    # Hyperparams
    SRC_VOCAB_SIZE = 32000
    TGT_VOCAB_SIZE = 32000
    D_MODEL = 256
    NUM_HEADS = 4
    D_FF = 1024
    NUM_LAYERS = 4
    DROPOUT = 0.1
    BATCH_SIZE = 16
    EPOCHS = 5
    LR = 1e-4
    MAX_SEQ_LEN = 128  # Set maximum sequence length

    # Create vocabularies
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    # First pass through data to build vocabulary
    with open("data/processed/train.en.bpe", 'r', encoding='utf-8') as f:
        for line in f:
            for token in line.strip().split():
                src_vocab.add_token(token)
    
    with open("data/processed/train.de.bpe", 'r', encoding='utf-8') as f:
        for line in f:
            for token in line.strip().split():
                tgt_vocab.add_token(token)
    
    # Create dataset with smaller max_len
    train_dataset = TranslationDataset(
        "data/processed/train.en.bpe", 
        "data/processed/train.de.bpe",
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=MAX_SEQ_LEN
    )
    
    # Create DataLoader with smaller batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=TranslationDataset.collate_fn
    )
    
    # Create model with smaller dimensions
    model = Transformer(
        len(src_vocab), 
        len(tgt_vocab), 
        d_model=D_MODEL, 
        num_heads=NUM_HEADS, 
        d_ff=D_FF, 
        num_layers=NUM_LAYERS, 
        dropout=DROPOUT
    ).to(device)
    
    # Initialize optimizer
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

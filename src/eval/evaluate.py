# src/eval/evaluate.py
import torch
from torch.utils.data import DataLoader
from src.models.transformer import Transformer
from src.train.train import TranslationDataset, Vocabulary
import sacrebleu
import sentencepiece as spm
from tqdm import tqdm
import os

def generate_subsequent_mask(size):
    # Returns shape (1, size, size) for easier broadcast
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask==1, float('-inf'))
    # shape => (size, size). Now unsqueeze(0) => (1, size, size)
    return mask.unsqueeze(0) 

def greedy_decode(model, src, device, max_len=50, start_symbol=1, end_symbol=2):
    model.eval()
    src = src.to(device)
    memory = model.encoder(src)  # (1, T_src, d_model) if batch=1
    ys = torch.ones(src.size(0), 1, dtype=torch.long, device=device).fill_(start_symbol)

    for i in range(max_len):
        T = ys.size(1)
        # Build a (T, T) causal mask, then expand to (1, 1, T, T)
        tgt_mask_2d = torch.triu(torch.ones(T, T), diagonal=1).bool().to(device)
        tgt_mask_4d = tgt_mask_2d.unsqueeze(0)  # (1, T, T)
        
        out = model.decoder(ys, memory, tgt_mask=tgt_mask_4d)
        next_word = torch.argmax(out[:, -1, :], dim=1).unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)

        if (next_word == end_symbol).all():
            break
    return ys

def decode_tokens(sp_model, tokens):
    """Convert token IDs back to text."""
    tokens = tokens.cpu().numpy().tolist()
    # Remove padding and special tokens
    tokens = [t for t in tokens if t > 2]  # Assuming 0=PAD, 1=SOS, 2=EOS
    return sp_model.decode(tokens)

def main():
    # Check for required files
    required_files = [
        "data/processed/sp_bpe.model",
        "data/processed/test.en.bpe",
        "data/processed/test.de.bpe",
        "checkpoints/transformer_base.pth"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file not found: {file}")

    # Setup device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    # Load SentencePiece models
    sp = spm.SentencePieceProcessor()
    sp.load("data/processed/sp_bpe.model")

    # ------------------------------------------------------------------
    # REBUILD THE SAME VOCABULARIES USED IN TRAINING
    # ------------------------------------------------------------------
    print("Rebuilding vocabulary from training data...")
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()

    train_en_path = "data/processed/train.en.bpe"
    train_de_path = "data/processed/train.de.bpe"
    
    with open(train_en_path, 'r', encoding='utf-8') as f:
        for line in f:
            for token in line.strip().split():
                src_vocab.add_token(token)
    
    with open(train_de_path, 'r', encoding='utf-8') as f:
        for line in f:
            for token in line.strip().split():
                tgt_vocab.add_token(token)

    actual_src_vocab_size = len(src_vocab)  # e.g., 15303
    actual_tgt_vocab_size = len(tgt_vocab)  # e.g., 22890
    
    print(f"Source vocab size: {actual_src_vocab_size}")
    print(f"Target vocab size: {actual_tgt_vocab_size}")

    # ------------------------------------------------------------------
    # CREATE THE TRANSFORMER MODEL WITH EXACT SAME DIMS AS TRAINING
    # ------------------------------------------------------------------
    model = Transformer(
        src_vocab_size=actual_src_vocab_size,
        tgt_vocab_size=actual_tgt_vocab_size,
        d_model=256,
        num_heads=4,
        d_ff=1024,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    # Load checkpoint
    ckpt_path = "checkpoints/transformer_base.pth"
    print(f"Loading model weights from {ckpt_path} ...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Prepare test set
    test_dataset = TranslationDataset("data/processed/test.en.bpe", "data/processed/test.de.bpe")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=TranslationDataset.collate_fn
    )

    references = []
    hypotheses = []

    print("Generating translations...")
    with torch.no_grad():
        for src, tgt in tqdm(test_loader, desc="Translating"):
            # Generate translation
            greedy_output = greedy_decode(model, src, device, max_len=100)
            
            # Convert token IDs back to text
            hypothesis = decode_tokens(sp, greedy_output[0])
            reference = decode_tokens(sp, tgt[0])
            
            hypotheses.append(hypothesis)
            references.append([reference])  # BLEU expects list of references

    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print(f"\nBLEU score: {bleu.score:.2f}")

    # Save translations
    with open("outputs/translations.txt", "w", encoding="utf-8") as f:
        for hyp, ref in zip(hypotheses, references):
            f.write(f"REF: {ref[0]}\n")
            f.write(f"HYP: {hyp}\n")
            f.write("-" * 80 + "\n")

if __name__ == "__main__":
    main()

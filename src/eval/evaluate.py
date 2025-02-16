# src/eval/evaluate.py
import torch
from torch.utils.data import DataLoader
from src.models.transformer import Transformer
from src.data.dataset import TranslationDataset
import sacrebleu
import sentencepiece as spm
from tqdm import tqdm
import os

def generate_subsequent_mask(size):
    """Generate mask for subsequent positions."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def greedy_decode(model, src, device, max_len=50, start_symbol=1, end_symbol=2):
    """
    Given a source sequence, generate a target sequence (greedy).
    """
    model.eval()
    src = src.to(device)
    memory = model.encoder(src)
    ys = torch.ones(src.size(0), 1, dtype=torch.long).fill_(start_symbol).to(device)

    for i in range(max_len):
        tgt_mask = generate_subsequent_mask(ys.size(1)).to(device)
        out = model.decoder(ys, memory, tgt_mask=tgt_mask)
        prob = out[:, -1, :]
        next_word = torch.argmax(prob, dim=1).unsqueeze(1)
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

    # Load model
    model_config = {
        'SRC_VOCAB_SIZE': 32000,
        'TGT_VOCAB_SIZE': 32000,
        'D_MODEL': 512,
        'NUM_HEADS': 8,
        'D_FF': 2048,
        'NUM_LAYERS': 6,
        'DROPOUT': 0.1
    }
    
    model = Transformer(
        model_config['SRC_VOCAB_SIZE'],
        model_config['TGT_VOCAB_SIZE'],
        model_config['D_MODEL'],
        model_config['NUM_HEADS'],
        model_config['D_FF'],
        model_config['NUM_LAYERS'],
        model_config['DROPOUT']
    ).to(device)
    
    model.load_state_dict(torch.load("checkpoints/transformer_base.pth", map_location=device))
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

# src/data/preprocess_data.py
import os
import glob
import sentencepiece as spm
from tqdm import tqdm

def combine_parallel_files(raw_dir, output_file):
    """
    Combine all downloaded WMT14 English and German files into a single file for BPE training.
    """
    print(f"Combining files into {output_file}")
    
    # Define the parallel files from our downloaded dataset
    parallel_files = [
        # Europarl
        ('europarl-v7.de-en.en', 'europarl-v7.de-en.de'),
        # Common Crawl
        ('commoncrawl.de-en.en', 'commoncrawl.de-en.de'),
        # News Commentary
        ('news-commentary-v9.de-en.en', 'news-commentary-v9.de-en.de')
    ]
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for en_file, de_file in parallel_files:
            en_path = os.path.join(raw_dir, en_file)
            de_path = os.path.join(raw_dir, de_file)
            
            if not (os.path.exists(en_path) and os.path.exists(de_path)):
                print(f"Warning: Could not find {en_file} or {de_file}")
                continue
                
            print(f"Processing {en_file} and {de_file}")
            
            # Add English text
            with open(en_path, 'r', encoding='utf-8') as f:
                outfile.write(f.read())
            
            # Add German text
            with open(de_path, 'r', encoding='utf-8') as f:
                outfile.write(f.read())

def train_sentencepiece(input_file, model_prefix, vocab_size=32000):
    """
    Train SentencePiece model on a combined text file of source + target.
    """
    print(f"Training SentencePiece model with vocab size {vocab_size}")
    spm.SentencePieceTrainer.Train(
        f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} "
        "--model_type=bpe --user_defined_symbols=<PAD>,<SOS>,<EOS>"
    )

def encode_with_sentencepiece(sp_model_path, input_file, output_file):
    """
    Encode a raw text file using the trained SentencePiece model.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            pieces = sp.encode(line.strip(), out_type=str)
            fout.write(" ".join(pieces) + "\n")

def split_parallel_data(raw_dir, output_dir, train_ratio=0.98, val_ratio=0.01):
    """
    Split the combined parallel data into train/validation/test sets.
    """
    print("Creating train/validation/test splits...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the parallel files from our downloaded dataset
    parallel_files = [
        ('europarl-v7.de-en.en', 'europarl-v7.de-en.de'),
        ('commoncrawl.de-en.en', 'commoncrawl.de-en.de'),
        ('news-commentary-v9.de-en.en', 'news-commentary-v9.de-en.de')
    ]
    
    # Collect all parallel sentences
    en_sentences, de_sentences = [], []
    for en_file, de_file in parallel_files:
        en_path = os.path.join(raw_dir, en_file)
        de_path = os.path.join(raw_dir, de_file)
        
        if not (os.path.exists(en_path) and os.path.exists(de_path)):
            print(f"Warning: Could not find {en_file} or {de_file}")
            continue
            
        with open(en_path, 'r', encoding='utf-8') as en_f, \
             open(de_path, 'r', encoding='utf-8') as de_f:
            for en_line, de_line in zip(en_f, de_f):
                en_sentences.append(en_line.strip())
                de_sentences.append(de_line.strip())
    
    # Calculate split sizes
    total_size = len(en_sentences)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Create splits
    splits = {
        'train': (0, train_size),
        'valid': (train_size, train_size + val_size),
        'test': (train_size + val_size, total_size)
    }
    
    # Write splits to files
    for split_name, (start, end) in splits.items():
        en_output = os.path.join(output_dir, f'{split_name}.en')
        de_output = os.path.join(output_dir, f'{split_name}.de')
        
        with open(en_output, 'w', encoding='utf-8') as en_f, \
             open(de_output, 'w', encoding='utf-8') as de_f:
            for en, de in zip(en_sentences[start:end], de_sentences[start:end]):
                en_f.write(en + '\n')
                de_f.write(de + '\n')
        
        print(f"Created {split_name} split with {end-start} sentence pairs")

def process_data(raw_dir="data/raw", processed_dir="data/processed", vocab_size=32000):
    """
    Complete data processing pipeline:
    1. Split data into train/valid/test
    2. Train SentencePiece model
    3. Encode all splits with SentencePiece
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create data splits
    split_parallel_data(raw_dir, processed_dir)
    
    # Combine all training data for BPE training
    combined_file = os.path.join(processed_dir, "combined_corpus.txt")
    with open(combined_file, 'w', encoding='utf-8') as outfile:
        for lang in ['en', 'de']:
            train_file = os.path.join(processed_dir, f'train.{lang}')
            with open(train_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
    
    # Train SentencePiece model
    model_prefix = os.path.join(processed_dir, "sp_bpe")
    train_sentencepiece(combined_file, model_prefix, vocab_size)
    
    # Encode all splits
    for split in ['train', 'valid', 'test']:
        for lang in ['en', 'de']:
            input_file = os.path.join(processed_dir, f'{split}.{lang}')
            output_file = os.path.join(processed_dir, f'{split}.{lang}.bpe')
            encode_with_sentencepiece(f"{model_prefix}.model", input_file, output_file)
    
    # Clean up combined corpus file
    os.remove(combined_file)
    print("\nData processing complete!")

if __name__ == "__main__":
    process_data()

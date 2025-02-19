# Transformer from Scratch

This project is an **open-source re-implementation** of the Transformer as described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). 

It aims to be:

- **Minimal**: The code closely follows the paper’s structure (multi-head attention, positional encodings, encoder-decoder design), providing an educational framework rather than a heavily optimized production library.
- **Modular**: Separate modules handle data preprocessing, model definition, training loops, and evaluation routines to promote clarity and extensibility.
- **Customizable**: Users can tweak hyperparameters (e.g., number of layers, hidden size, dropout) to replicate “Base,” “Big,” or smaller variants of the Transformer.  
- **Practical**: Includes end-to-end scripts for downloading and preprocessing data (e.g., WMT14), training on Apple Silicon (M-series) hardware, and evaluating the model with BLEU scoring.  

## Overview

The transformer is a neural sequence-to-sequence architecture for sequence transduction tasks such as machine translation. Unlike approaches that rely on recurrent or convolutional layers, the Transformer uses self-attention mechanisms to model long-range dependencies efficiently in both the encoder and decoder. 

Key contributions of the Transformer include:

1. **Scaled Dot-Product Attention**: A mechanism that computes pairwise interactions among tokens with computational cost proportional only to the product of sequence length and embedding dimensionality.  
2. **Multi-Head Attention**: Multiple independent attention heads capture different aspects or subspaces of a given representation.  
3. **Positional Encodings**: A way to inject sequence-order information into the model, compensating for the lack of recurrent or convolutional structures.

Empirically, the Transformer achieved state-of-the-art performance on WMT machine translation benchmarks and significantly reduced training time through parallelization.

## Requirements

- Python 3.10+
- PyTorch 2.6.0+ (with MPS support for M3 Max)
- NumPy
- tqdm

```bash
conda env create -f environment.yml
conda activate transformer_m3
```

## Project Structure

```bash
transformer-from-scratch/
├── src/
│ ├── data/
│ │ ├── __init__.py
│ │ ├── download_data.py
│ │ └── preprocess_data.py
│ ├── model/
│ │ ├── __init__.py
│ │ └── transformer.py
│ ├── train/
│ │ ├── __init__.py
│ │ └── train.py
│ ├── eval/
│ │ ├── __init__.py
│ │ └── evaluate.py
│ ├── utils/
│ │ └── __init__.py
│ └── train.py
├── environment.yml
├── LICENSE
└── README.md
```

## Usage

### Basic Example

```python

from src.models.transformer import Transformer

# Initialize the model
model = Transformer(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    d_model=512,
    num_heads=8,
    num_layers=6,
)
```

# Training example 

## Downloading & preprocessing the data

```bash
python src/data/download_data.py
python src/data/preprocess_data.py
```

## Training

To train the model:

```bash

python src.train.train
```

## Evaluating the model

To evaluate the model:

```bash

python src.eval.evaluate
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{vaswani2017attention,
title={Attention is all you need},
author={Vaswani, Ashish and others},
journal={Advances in neural information processing systems},
volume={30},
year={2017}
} 
```

## Contact

For questions or feedback, please open an issue in the repository.


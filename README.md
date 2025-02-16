# Transformer from Scratch

This project implements the Transformer architecture as described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). The implementation is optimized for Mac M3 Max machines using PyTorch.

## Overview

The Transformer is a revolutionary architecture that relies entirely on self-attention mechanisms to compute representations of its input and output, replacing traditional recurrent neural networks. This implementation provides a clear, educational approach to understanding the core concepts.

## Requirements

- Python 3.9+
- PyTorch 2.0+ (with MPS support for M3 Max)
- NumPy
- tqdm
- matplotlib (for visualization)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/mps
```

## Project Structure

```bash
transformer-from-scratch/
├── src/
│ ├── model/
│ │ ├── attention.py
│ │ ├── encoder.py
│ │ ├── decoder.py
│ │ └── transformer.py
│ ├── utils/
│ │ ├── data_loader.py
│ │ └── visualization.py
│ └── train.py
├── tests/
├── notebooks/
│ └── transformer_demo.ipynb
└── README.md
```

## Implementation Details

The implementation includes the following key components:

1. **Multi-Head Attention**
   - Scaled dot-product attention
   - Parallel attention heads
   - Linear projections for queries, keys, and values

2. **Position-wise Feed-Forward Networks**
   - Two linear transformations with ReLU activation

3. **Positional Encoding**
   - Sinusoidal position encoding
   - Learnable position embeddings (optional)

4. **Layer Normalization and Residual Connections**

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

### Using MPS Acceleration

This implementation is optimized for Mac M3 Max using PyTorch's MPS (Metal Performance Shaders) backend:

```python
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
```


## Training

To train the model:

```bash

python src/train.py --batch_size 32 --epochs 100 --lr 0.0001
```


## Performance

On a Mac M3 Max, you can expect the following performance metrics:
- Training speed: ~X samples/second
- Memory usage: ~Y GB
- Training time for standard translation task: ~Z hours

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


## Acknowledgments

- The original Transformer paper authors
- PyTorch team for MPS support
- Open-source community

## Contact

For questions or feedback, please open an issue in the repository.


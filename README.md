# TimeVQGAN: Universal Time-series Vector Quantized Generative Adversarial Networks

Official implementation of "TimeVQGAN: Universal Time-series Vector Quantized Generative Adversarial Networks" (ICDE 2026).

TimeVQGAN converts continuous time series into discrete token sequences via vector quantization and adversarial training, enabling seamless integration with large language models and other token-based architectures.

## Features

- **Universal Tokenization**: Converts arbitrary time series into discrete tokens (block size 4)
- **Large-Scale Pretraining**: Pretrained on Time-300B dataset with 16,384 codebook size
- **State-of-the-Art Performance**: Achieves SOTA results across 8 tasks including forecasting, classification, anomaly detection, and generation

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- NVIDIA GPU with CUDA
- PyTorch 2.0+

**Download pretrained models:**
- TimeVQGAN checkpoint: [Baidu Netdisk](https://pan.baidu.com/s/1sS5JTyOp82YJL04CRmM50w?pwd=sj1w) → `checkpoints/TimeVQGAN.pt`
- TimeMoE backbone: [HuggingFace](https://huggingface.co/Maple728/TimeMoE-50M) → `TimeMoE_50M/`

## Quick Start

Run the demo:
```bash
python TimeVQGAN_tokenizer.py
```

Programmatic usage:
```python
import torch
from TimeVQGAN_tokenizer import TimeVQGAN_Tokenizer

# Initialize tokenizer
tokenizer = TimeVQGAN_Tokenizer().cuda()

# Encode time series to discrete codes
time_series = torch.randn(2048).cuda()
codes = tokenizer.encode(time_series)

# Decode back to continuous signal
reconstructed = tokenizer.decode(codes)
```

## Repository Structure

```
├── TimeVQGAN.py              # Core model components
├── TimeVQGAN_tokenizer.py    # Tokenizer API and demo
├── checkpoints/              # Pretrained weights (download required)
├── TimeMoE_50M/             # TimeMoE backbone (download required)
└── time_moe/                # TimeMoE utilities
```



**Note:** Full training scripts and MoVE implementation will be released soon.

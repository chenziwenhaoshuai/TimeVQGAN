# TimeVQGAN

TimeVQGAN is a minimal, self-contained implementation of a Vector-Quantized GAN for time series built on top of a TimeMoE encoder/decoder stack. It provides:

- An encoder that downsamples and embeds 1D signals with a pretrained TimeMoE model
- A vector-quantized codebook for discrete tokenization of time series
- A transformer-based decoder to reconstruct signals from code indices
- A simple perceptual loss module for training (LPIPS-style over hidden states)


## Repository structure

- `TimeVQGAN.py` — Core model components: encoder (TimeMoE-based), codebook, decoder, LPIPS module, and the TimeVQGAN forward/encode/decode methods
- `TimeVQGAN_tokenizer.py` — A small wrapper exposing an encode/decode API for time series and a demo that visualizes reconstruction
- `checkpoints/TimeVQGAN.pt` — Pretrained VQ-GAN checkpoint (used by `TimeVQGAN_tokenizer.py`)
- `TimeMoE_50M/` — A local Hugging Face-style folder with config and `model.safetensors` weights for the TimeMoE backbone used by the encoder
- `time_moe/` — Minimal TimeMoE implementation and training utilities


## Requirements

- Python 3.9+ (tested with 3.9)
- NVIDIA GPU + CUDA required to run the provided code as-is
  - The encoder and parts of the pipeline call `.cuda()` directly; CPU-only usage would require minor code changes

Install Python dependencies via `requirements.txt` (see below for notes on PyTorch and optional extras):

```
pip install -r requirements.txt
```

Notes:
- Install PyTorch matching your CUDA version from https://pytorch.org/get-started/locally/ if the default pip wheel doesn’t match your system.
- FlashAttention is optional and Linux-only; the code safely falls back when it’s unavailable.
- Deepspeed is optional and only needed if you enable it in training arguments.


## Quickstart: tokenize and reconstruct a time series

The simplest way to try the tokenizer is to run the demo in `TimeVQGAN_tokenizer.py` which:
- Builds the tokenizer on GPU
- Encodes a synthetic 1D signal into discrete code indices
- Decodes it back to a reconstructed signal
- Plots the original and reconstruction

Windows (cmd.exe):

```
python TimeVQGAN_tokenizer.py
```

You should see the printed code indices and a Matplotlib window with two subplots (original vs reconstruction).

## Programmatic usage

Example: turn an arbitrary 1D float tensor into code indices and back.

```python
import torch
from TimeVQGAN_tokenizer import TimeVQGAN_Tokenizer

# 1) Prepare a 1D float tensor on GPU (length is arbitrary)
#    The tokenizer will chunk/pad internally to 512-length blocks.
series = torch.sin(torch.linspace(0, 200, 4096)).cuda()

# 2) Build tokenizer (loads TimeVQGAN and pretrained weights)
model = TimeVQGAN_Tokenizer().cuda()

# 3) Encode -> 1D long tensor of code indices
codes = model.encode(series)

# 4) Decode -> reconstructed 1D float tensor
recon = model.decode(codes)
```


## Training and datasets

The training script will be open-sourced later.

## License

No license file was found in this repository. If you plan to redistribute or publish results, please add an appropriate license file and ensure you have the rights to the included weights and datasets.


## Citation

If you use this codebase in your research, please consider citing relevant VQ-GAN and MoE/Time-series transformer works. Add your own bib entries here.


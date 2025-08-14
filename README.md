# LLM

# Advanced Code Generation Transformer

A PyTorch GPT-style transformer for generating code using a token-level BPE tokenizer. Supports mixed precision training, causal self-attention, and checkpointing.

---

## Features

- Token-level code generation with Byte-Pair Encoding (BPE)
- Multi-layer GPT transformer with causal self-attention
- Autoregressive generation
- Mixed precision (FP16) training
- Gradient clipping and learning rate scheduling
- Checkpoint saving and resuming

---

## Installation

```bash
git clone https://github.com/yourusername/codegen-transformer-advanced.git
cd codegen-transformer-advanced
pip install -r requirements.txt

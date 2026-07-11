# LLM

# Advanced Code Generation Transformer

A PyTorch GPT-style transformer for generating code using a token-level BPE tokenizer. Supports mixed precision training, causal self-attention, and checkpointing.

---

## Features

- Token-level code generation with Byte-Pair Encoding (BPE)
- Multi-layer GPT transformer with causal self-attention
- Autoregressive generation
- Mixed precision  training
- Gradient clipping and learning rate scheduling
- Checkpoint saving and resuming

---

## Installation

```bash
git clone https://github.com/Oliprg3/LLM.git
cd LLM
pip install -r requirements.txt

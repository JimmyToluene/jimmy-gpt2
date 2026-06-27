# jimmy-gpt2

A from-scratch reimplementation of a **GPT-2 style transformer** in PyTorch, built as a
learning project to understand the internals of modern language models.

## What's inside

I implement the core transformer building blocks step by step and verify each one with
quick shape checks before assembling them into a full GPT model.

- **Causal Self-Attention** — multi-head masked attention (QKV projection, per-head split, causal mask, softmax, output projection)
- **MLP** — position-wise feed-forward network with GELU activation
- **Block** — a transformer block combining attention + MLP with pre-LayerNorm residual connections
- **GPT** — the full model: token + positional embeddings, a stack of transformer blocks, final LayerNorm, and the language-model head

## Project structure

```
.
├── train_gpt2.py                       # Full GPT-2 model definition (attention, MLP, block, GPT)
└── notebooks/
    ├── playground.ipynb                # Step-by-step walkthrough of each module with shape checks
    └── pre-trained_playground.ipynb    # Experiments with a pre-trained GPT-2
```

## Model configuration

The default config is a small GPT-2 variant:

| Parameter    | Value |
| ------------ | ----- |
| `block_size` | 256   |
| `vocab_size` | 65    |
| `n_layer`    | 6     |
| `n_head`     | 6     |
| `n_embd`     | 384   |

## Getting started

```bash
# Requires Python 3 and PyTorch
pip install torch

# Explore the building blocks interactively
jupyter notebook notebooks/playground.ipynb
```

## Acknowledgements

Inspired by Andrej Karpathy's "Let's build GPT" / nanoGPT walkthroughs.

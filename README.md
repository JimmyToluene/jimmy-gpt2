# jimmy-gpt2

A from-scratch reimplementation of the **GPT-2** transformer in PyTorch, built as a
learning project to understand the internals of modern language models. It can
load OpenAI's pretrained GPT-2 weights to generate text, and run a training
forward pass that computes the language-modeling loss.

## What's inside

I implement the core transformer building blocks step by step, verify each one with
quick shape checks, then assemble them into a full GPT model that is weight-compatible
with HuggingFace's GPT-2.

### Model architecture

The building blocks, composed bottom-up into the full model:

- **Causal Self-Attention**: multi-head masked attention (QKV projection, per-head split, causal mask, softmax, output projection)
- **MLP**: position-wise feed-forward network with GELU activation
- **Block**: a transformer block combining attention and MLP with pre-LayerNorm residual connections
- **GPT**: the full model, with token and positional embeddings, a stack of transformer blocks, a final LayerNorm, and the language-model head

### Training & inference

- **Forward pass / loss**: `GPT.forward(idx, target)` returns logits, and optionally computes the cross-entropy loss for next-token prediction when targets are supplied
- **Text generation**: top-k sampling loop that autoregressively extends a prompt

### Pretrained weights

- **`GPT.from_pretrained(...)`**: loads pretrained weights from HuggingFace (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`) into the from-scratch model

### Data & utilities

- **Data batching**: tokenizes a plain-text corpus (`datasets/input.txt`, Tiny Shakespeare) with `tiktoken`'s GPT-2 encoder and builds `(input, target)` batches for next-token prediction
- **Device autodetection**: picks CUDA, Apple MPS, or CPU automatically

## Project structure

```
.
├── train_gpt2.py                       # Full GPT-2 model + pretrained weight loading + forward-pass loss + text generation
├── datasets/
│   └── input.txt                       # Training corpus (Tiny Shakespeare)
└── notebooks/
    ├── playground.ipynb                # Step-by-step walkthrough of each module with shape checks
    └── pre-trained_playground.ipynb    # Experiments with a pretrained GPT-2
```

## Model configuration

The default config matches the smallest GPT-2 (124M):

| Parameter    | Value | Notes                                            |
| ------------ | ----- | ------------------------------------------------ |
| `block_size` | 1024  | context window size                              |
| `vocab_size` | 50257 | 50,000 BPE merges + 256 byte tokens + 1 EOS      |
| `n_layer`    | 12    | number of transformer blocks                     |
| `n_head`     | 12    | number of attention heads                        |
| `n_embd`     | 768   | embedding dimension                              |

Larger variants (`gpt2-medium/large/xl`) are configured automatically by
`GPT.from_pretrained`.

## Getting started

```bash
# Requires Python 3; runs on CPU, and uses CUDA or Apple MPS automatically when available
pip install torch transformers tiktoken

# Build a data batch and run a forward pass that prints the loss
python train_gpt2.py

# Or explore the building blocks interactively
jupyter notebook notebooks/playground.ipynb
```

## Acknowledgements

Inspired by Andrej Karpathy's "Let's build GPT" / nanoGPT walkthroughs.

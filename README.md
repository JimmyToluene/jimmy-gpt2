# jimmy-gpt2

A from-scratch reimplementation of the **GPT-2** transformer in PyTorch, built as a
learning project to understand the internals of modern language models — and able to
load OpenAI's pretrained GPT-2 weights to generate text.

## What's inside

I implement the core transformer building blocks step by step, verify each one with
quick shape checks, then assemble them into a full GPT model that is weight-compatible
with HuggingFace's GPT-2.

- **Causal Self-Attention** — multi-head masked attention (QKV projection, per-head split, causal mask, softmax, output projection)
- **MLP** — position-wise feed-forward network with GELU activation
- **Block** — a transformer block combining attention + MLP with pre-LayerNorm residual connections
- **GPT** — the full model: token + positional embeddings, a stack of transformer blocks, final LayerNorm, and the language-model head
- **`GPT.from_pretrained(...)`** — loads pretrained weights from HuggingFace (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`) into the from-scratch model

## Project structure

```
.
├── train_gpt2.py                       # Full GPT-2 model + pretrained weight loading + text generation
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
# Requires Python 3, with a CUDA-capable GPU for generation
pip install torch transformers tiktoken

# Load pretrained GPT-2 and set up generation
python train_gpt2.py

# Or explore the building blocks interactively
jupyter notebook notebooks/playground.ipynb
```

## Acknowledgements

Inspired by Andrej Karpathy's "Let's build GPT" / nanoGPT walkthroughs.

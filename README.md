# PicoGPT: A NumPy Implementation of GPT-2

A minimal, educational implementation of GPT-2 using only NumPy and raw linear algebra. This project helps you understand the Transformer architecture from the ground up.

## Overview

PicoGPT implements the GPT-2 architecture (124M parameters) using only NumPy, focusing on the **forward pass** (inference) rather than training. This "Inference-First" approach allows you to:

- Understand the mathematical foundations of Transformers
- See how attention mechanisms work at the matrix level
- Generate text using pre-trained GPT-2 weights
- Learn without the complexity of backpropagation and GPU setup

## Project Structure

```
uw-msis-gpt-lab/
├── pico_gpt.py              # Main implementation (all core functions)
├── setup_weights.py          # Script to download and convert GPT-2 weights
├── web_ui.py                 # Streamlit web interface (optional)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── instructions/             # Educational materials
│   ├── student_builder_prompts.md  # Prompts for building with AI assistants
│   ├── architecture_visualization.html  # Interactive architecture diagram
│   ├── pico_gpt_architecture.md  # Code-to-concept mapping guide
│   └── pico_gpt.py          # Original file (reference)
└── tests/                    # Unit tests
    └── test_pico_gpt.py     # Test suite
```

## Quick Start - Run Locally

Follow these simple steps to run PicoGPT on your local machine:

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Make sure you have Python 3.7 or higher installed.

### Step 2: Download GPT-2 Weights (One-time Setup)

Run this command **once** to download and convert the pre-trained GPT-2 weights:

```bash
python setup_weights.py
```

This will download ~500MB of weights and save them as `gpt2_weights.npz` in the project directory.

### Step 3: Run the Model

**Option A: Command Line Interface**

```bash
python pico_gpt.py
```

This will generate text from the default prompt: "Alan Turing theorized that computers would one day become"

To use a custom prompt, edit `pico_gpt.py` and change the `prompt` variable around line 374:

```python
prompt = "Your custom prompt here"
```

**Option B: Web Interface (Recommended)**

For an interactive experience with a user-friendly interface:

```bash
streamlit run web_ui.py
```

This will:
- Open a web browser automatically (usually at `http://localhost:8501`)
- Allow you to enter custom prompts directly in the UI
- Let you adjust generation parameters (temperature, tokens, etc.)
- Show real-time generation progress

**Note**: The web UI requires Streamlit (already included in requirements.txt).

### What You Can Control in the Web UI

- **Prompt**: Enter any text to generate from
- **Tokens to Generate**: Number of tokens to generate (1-50)
- **Temperature** (0.1-2.0): Controls randomness - lower = more focused, higher = more creative
- **Top-K**: Number of top tokens to consider for sampling
- **Top-P**: Nucleus sampling threshold
- **Frequency Penalty** (0.0-2.0): Reduces repetition by penalizing recently used tokens

### Troubleshooting

- **"Weights not found" error**: Make sure you ran `python setup_weights.py` first
- **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Port already in use** (web UI): Streamlit will try a different port automatically, or you can specify one with `streamlit run web_ui.py --server.port 8502`

## Architecture Overview

PicoGPT implements the following components:

1. **Activation Functions**: GELU (Gaussian Error Linear Unit)
2. **Normalization**: Layer Normalization
3. **Attention**: Scaled Dot-Product Attention with causal masking
4. **Multi-Head Attention**: Parallel attention heads
5. **Feed-Forward Network**: Two-layer MLP with GELU activation
6. **Transformer Block**: Attention + FFN with residual connections
7. **GPT-2 Model**: 12 transformer blocks with embeddings

## Key Functions

- `gelu(x)`: GELU activation function
- `softmax(x)`: Softmax normalization
- `layer_norm(x, g, b)`: Layer normalization
- `linear(x, w, b)`: Linear transformation (xW + b)
- `attention(q, k, v, mask)`: Scaled dot-product attention
- `mha(x, c_attn, c_proj, n_head)`: Multi-head attention
- `feed_forward(x, c_fc, c_proj)`: Position-wise feed-forward network
- `transformer_block(...)`: Complete transformer block
- `gpt2(...)`: Full GPT-2 model forward pass
- `generate(...)`: Autoregressive text generation loop

## Building from Scratch

If you want to build PicoGPT yourself and understand every component, check out these resources:

- **LEARNING_GUIDE.md**: Comprehensive step-by-step guide to build PicoGPT from scratch
- **instructions/student_builder_prompts.md**: Prompts to use with AI assistants when you get stuck
- **instructions/architecture_visualization.html**: Interactive visual diagram of the GPT-2 architecture
- **instructions/pico_gpt_architecture.md**: Code-to-concept mapping guide that explains how each function maps to Transformer concepts

## Running Tests

```bash
python -m pytest tests/test_pico_gpt.py -v
```

## Requirements

- Python 3.7+
- NumPy
- PyTorch (for downloading weights)
- Transformers (Hugging Face)
- tqdm (for progress bars)
- Streamlit (for web UI, optional)

## Notes

- This implementation is **inference-only** (no training)
- Uses **greedy sampling** (always picks most likely token)
- For production systems, consider implementing nucleus sampling (Top-P)
- The model uses pre-trained GPT-2 weights from Hugging Face

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## License

Educational use only. GPT-2 weights are subject to OpenAI's usage policy.


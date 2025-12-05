# PicoGPT: A NumPy Implementation of GPT-2

A minimal, educational implementation of GPT-2 using only NumPy and raw linear algebra. This project is designed for MSIS students to understand the Transformer architecture from the ground up.

## Overview

PicoGPT implements the GPT-2 architecture (124M parameters) using only NumPy, focusing on the **forward pass** (inference) rather than training. This "Inference-First" approach allows students to:

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
│   ├── instructions.md       # Instructor prompts for teaching
│   ├── student_builder_prompts.md  # Prompts for students
│   ├── strategy.txt          # Project thesis and rationale
│   └── pico_gpt.py          # Original file (reference)
└── tests/                    # Unit tests
    └── test_pico_gpt.py     # Test suite
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download GPT-2 Weights

Run this script **once** to download and convert GPT-2 weights:

```bash
python setup_weights.py
```

This will create `gpt2_weights.npz` (~500MB) containing the pre-trained weights in NumPy format.

### 3. Run the Model

```bash
python pico_gpt.py
```

This will generate text from the default prompt: "Alan Turing theorized that computers would one day become"

### 4. Customize the Prompt

Edit `pico_gpt.py` and modify the `prompt` variable in the `__main__` section:

```python
prompt = "Your custom prompt here"
```

### 5. (Optional) Run the Web Interface

For a more interactive experience, use the Streamlit web UI:

```bash
streamlit run web_ui.py
```

This will open a web browser with a user-friendly interface where you can:
- Enter custom prompts
- Adjust the number of tokens to generate
- See real-time generation progress
- View the top 5 predictions for each token

**Note**: The web UI requires Streamlit (already included in requirements.txt).

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

## Educational Resources

### For Students

- **LEARNING_GUIDE.md**: Comprehensive step-by-step guide to build PicoGPT from scratch
- **instructions/student_builder_prompts.md**: Prompts to use with AI assistants when stuck
- **instructions/architecture_visualization.html**: Interactive visual diagram of the GPT-2 architecture
- **instructions/pico_gpt_architecture.md**: Code-to-concept mapping guide

### For Instructors

- **instructions/instructions.md**: Prompts for instructors to help students understand linear algebra concepts
- **instructions/strategy.txt**: Project rationale and pedagogical approach
- **instructions/pico_gpt_architecture.md**: Visual mapping of functions to concepts

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


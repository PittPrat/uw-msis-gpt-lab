# PicoGPT Learning Guide

A step-by-step guide to building and understanding GPT-2 from scratch using NumPy.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Building Step-by-Step](#building-step-by-step)
4. [Testing Your Implementation](#testing-your-implementation)
5. [Common Pitfalls & Debugging](#common-pitfalls--debugging)
6. [Exercises & Challenges](#exercises--challenges)
7. [Resources](#resources)

---

## Getting Started

### Prerequisites

Before you begin, make sure you understand:
- **Python basics**: Functions, lists, dictionaries
- **NumPy basics**: Arrays, matrix multiplication (`@`), reshaping
- **Linear algebra**: Matrix multiplication, transpose, dot product
- **Neural networks**: Basic concepts (optional but helpful)

### Setup

1. **Clone/Download the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the weights** (one-time setup):
   ```bash
   python setup_weights.py
   ```

### Your Goal

By the end of this guide, you will have built a working GPT-2 implementation that can generate text, understanding every line of code you write.

---

## Understanding the Architecture

### The Big Picture

GPT-2 is a **transformer-based language model**. Here's what it does:

```
Input Text → Tokenize → Embeddings → Transformer Blocks → Output Logits → Generate Next Word
```

### Key Components (Build Order)

1. **Basic Functions** (gelu, softmax, layer_norm, linear)
2. **Attention Mechanism** (attention, mha)
3. **Feed-Forward Network** (feed_forward)
4. **Transformer Block** (transformer_block)
5. **Full Model** (gpt2)
6. **Generation Loop** (generate)

### Why This Order?

We build from the bottom up:
- Start with simple math operations
- Combine them into attention
- Combine attention into transformer blocks
- Stack blocks into the full model
- Add generation logic on top

---

## Building Step-by-Step

### Phase 1: Basic Building Blocks

#### Step 1.1: Implement `softmax()`

**What it does**: Converts logits (raw scores) into probabilities (0-1, sum to 1)

**Why it matters**: Every neural network needs this to make predictions

**Your task**:
```python
def softmax(x):
    """
    Convert logits to probabilities.
    Hint: Use np.exp() and np.sum()
    """
    # TODO: Implement this
    # Steps:
    # 1. Subtract max for numerical stability
    # 2. Apply exp()
    # 3. Normalize by sum
    pass
```

**Test it**:
```python
x = np.array([1.0, 2.0, 3.0])
result = softmax(x)
print(result)  # Should sum to ~1.0
print(np.sum(result))  # Should be 1.0
```

**Check your answer**: Each value should be between 0 and 1, and they should sum to 1.

---

#### Step 1.2: Implement `gelu()`

**What it does**: Activation function (like ReLU, but smoother)

**Why it matters**: Adds non-linearity so the model can learn complex patterns

**Your task**:
```python
def gelu(x):
    """
    Gaussian Error Linear Unit activation.
    Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    # TODO: Implement this
    pass
```

**Test it**:
```python
x = np.array([-2.0, 0.0, 2.0])
result = gelu(x)
print(result)  # Should handle negative values smoothly
```

**Check your answer**: 
- gelu(0) should be approximately 0
- gelu(positive) should be positive
- gelu(negative) should be negative but closer to 0

---

#### Step 1.3: Implement `layer_norm()`

**What it does**: Normalizes inputs to have mean=0, variance=1

**Why it matters**: Stabilizes training and helps gradients flow

**Your task**:
```python
def layer_norm(x, g, b, eps=1e-5):
    """
    Layer normalization.
    x: input [..., features]
    g: gamma (scale) [features]
    b: beta (shift) [features]
    """
    # TODO: Implement this
    # Steps:
    # 1. Calculate mean
    # 2. Calculate variance
    # 3. Normalize: (x - mean) / sqrt(variance + eps)
    # 4. Scale and shift: g * normalized + b
    pass
```

**Test it**:
```python
x = np.random.randn(5, 10)
g = np.ones(10)
b = np.zeros(10)
result = layer_norm(x, g, b)
print(np.mean(result, axis=-1))  # Should be ~0
print(np.var(result, axis=-1))   # Should be ~1
```

**Check your answer**: After normalization with g=1, b=0, mean should be ~0 and variance should be ~1.

---

#### Step 1.4: Implement `linear()`

**What it does**: Matrix multiplication with bias (y = xW + b)

**Why it matters**: This is the fundamental operation of neural networks

**Your task**:
```python
def linear(x, w, b):
    """
    Linear transformation: y = xW + b
    x: [batch, in_features] or [in_features]
    w: [in_features, out_features]
    b: [out_features]
    """
    # TODO: Implement this
    # Hint: Use @ for matrix multiplication
    pass
```

**Test it**:
```python
x = np.array([[1.0, 2.0]])
w = np.array([[1.0, 0.0], [0.0, 1.0]])
b = np.array([1.0, 1.0])
result = linear(x, w, b)
print(result)  # Should be [[2.0, 3.0]]
```

**Check your answer**: This is just matrix multiplication: `x @ w + b`

---

### Phase 2: Attention Mechanism

#### Step 2.1: Understand Attention Concept

**The Analogy**: Think of attention like a librarian searching a database:
- **Query (Q)**: "What are you looking for?"
- **Key (K)**: "What's in each book?"
- **Value (V)**: "The actual content"

The model "attends" to different parts of the input based on relevance.

#### Step 2.2: Implement `attention()`

**What it does**: Computes how much each token should "pay attention" to other tokens

**The Formula**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

**Your task**:
```python
def attention(q, k, v, mask):
    """
    Scaled Dot-Product Attention.
    q: [n_heads, seq_len, head_dim]
    k: [n_heads, seq_len, head_dim]
    v: [n_heads, seq_len, head_dim]
    mask: [seq_len, seq_len] (causal mask)
    """
    # TODO: Implement this
    # Steps:
    # 1. Compute Q @ K^T (transpose last two dims of K)
    # 2. Scale by sqrt(d_k) where d_k = q.shape[-1]
    # 3. Add mask (expand mask to match attention_scores shape)
    # 4. Apply softmax
    # 5. Multiply by V
    pass
```

**Key Questions to Answer**:
1. Why do we transpose K? (Hint: matrix multiplication dimensions)
2. Why divide by sqrt(d_k)? (Hint: prevents large values)
3. What does the mask do? (Hint: prevents looking at future tokens)

**Test it**:
```python
n_heads, seq_len, head_dim = 2, 3, 4
q = np.random.randn(n_heads, seq_len, head_dim)
k = np.random.randn(n_heads, seq_len, head_dim)
v = np.random.randn(n_heads, seq_len, head_dim)
mask = (1 - np.tri(seq_len)) * -1e10
result = attention(q, k, v, mask)
print(result.shape)  # Should be [n_heads, seq_len, head_dim]
```

---

#### Step 2.3: Implement `mha()` (Multi-Head Attention)

**What it does**: Runs multiple attention "heads" in parallel, each focusing on different relationships

**Why multiple heads**: One head might focus on grammar, another on meaning, another on syntax

**Your task**:
```python
def mha(x, c_attn, c_proj, n_head):
    """
    Multi-Head Attention.
    x: [seq_len, n_embd]
    c_attn: {'w': [n_embd, 3*n_embd], 'b': [3*n_embd]}
    c_proj: {'w': [n_embd, n_embd], 'b': [n_embd]}
    n_head: number of attention heads
    """
    # TODO: Implement this
    # Steps:
    # 1. Project to Q, K, V using linear() with c_attn
    # 2. Split into Q, K, V (use np.split)
    # 3. Reshape for heads: [seq_len, n_head, head_dim]
    # 4. Transpose to [n_head, seq_len, head_dim]
    # 5. Create causal mask
    # 6. Call attention()
    # 7. Merge heads back: transpose and reshape
    # 8. Project output with linear() using c_proj
    pass
```

**Key Challenge**: Understanding the reshaping! 
- Input: `[seq_len, n_embd]` where `n_embd = n_head * head_dim`
- After QKV projection: `[seq_len, 3*n_embd]`
- After split: Three arrays of `[seq_len, n_embd]`
- Reshape each: `[seq_len, n_head, head_dim]`
- Transpose: `[n_head, seq_len, head_dim]`

**Test it**:
```python
seq_len, n_embd, n_head = 5, 12, 3
x = np.random.randn(seq_len, n_embd)
c_attn = {'w': np.random.randn(n_embd, 3*n_embd), 'b': np.random.randn(3*n_embd)}
c_proj = {'w': np.random.randn(n_embd, n_embd), 'b': np.random.randn(n_embd)}
result = mha(x, c_attn, c_proj, n_head)
print(result.shape)  # Should be [seq_len, n_embd]
```

---

### Phase 3: Feed-Forward Network

#### Step 3.1: Implement `feed_forward()`

**What it does**: A simple 2-layer neural network applied to each token independently

**Structure**: Expand → Activate → Contract

**Your task**:
```python
def feed_forward(x, c_fc, c_proj):
    """
    Position-wise Feed-Forward Network.
    x: [seq_len, n_embd]
    c_fc: {'w': [n_embd, 4*n_embd], 'b': [4*n_embd]}
    c_proj: {'w': [4*n_embd, n_embd], 'b': [n_embd]}
    """
    # TODO: Implement this
    # Steps:
    # 1. Expand: linear(x, c_fc['w'], c_fc['b'])
    # 2. Activate: gelu(expanded)
    # 3. Contract: linear(activated, c_proj['w'], c_proj['b'])
    pass
```

**Why expand then contract?**: 
- Expansion allows the model to learn complex transformations
- Contraction brings it back to the original dimension
- This pattern is common in transformers

**Test it**:
```python
seq_len, n_embd = 5, 10
x = np.random.randn(seq_len, n_embd)
c_fc = {'w': np.random.randn(n_embd, 4*n_embd), 'b': np.random.randn(4*n_embd)}
c_proj = {'w': np.random.randn(4*n_embd, n_embd), 'b': np.random.randn(n_embd)}
result = feed_forward(x, c_fc, c_proj)
print(result.shape)  # Should be [seq_len, n_embd]
```

---

### Phase 4: Transformer Block

#### Step 4.1: Implement `transformer_block()`

**What it does**: Combines attention and feed-forward with residual connections

**Structure**:
```
x = x + MHA(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Why residual connections?**: They allow gradients to flow through deep networks

**Your task**:
```python
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    """
    A Single Transformer Block.
    x: [seq_len, n_embd]
    mlp: feed-forward parameters
    attn: attention parameters
    ln_1: first layer norm {'g': [n_embd], 'b': [n_embd]}
    ln_2: second layer norm {'g': [n_embd], 'b': [n_embd]}
    n_head: number of attention heads
    """
    # TODO: Implement this
    # Steps:
    # 1. Attention with residual: x = x + mha(layer_norm(x, **ln_1), ...)
    # 2. Feed-forward with residual: x = x + feed_forward(layer_norm(x, **ln_2), ...)
    pass
```

**Test it**:
```python
# Create dummy parameters (you'll load real ones later)
seq_len, n_embd, n_head = 5, 12, 3
x = np.random.randn(seq_len, n_embd)
# ... create all parameter dictionaries ...
result = transformer_block(x, mlp, attn, ln_1, ln_2, n_head)
print(result.shape)  # Should be [seq_len, n_embd]
```

---

### Phase 5: Full GPT-2 Model

#### Step 5.1: Implement `gpt2()`

**What it does**: The complete GPT-2 architecture

**Structure**:
1. Token embeddings + Positional embeddings
2. 12 Transformer blocks
3. Final layer norm
4. Project to vocabulary

**Your task**:
```python
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    """
    The Full GPT-2 Model Architecture.
    inputs: list of token IDs [int, int, ...]
    wte: token embeddings [vocab_size, n_embd]
    wpe: positional embeddings [max_seq_len, n_embd]
    blocks: list of transformer block parameters
    ln_f: final layer norm {'g': [n_embd], 'b': [n_embd]}
    n_head: number of attention heads
    """
    # TODO: Implement this
    # Steps:
    # 1. Look up token embeddings: wte[inputs]
    # 2. Add positional embeddings: + wpe[range(len(inputs))]
    # 3. Pass through each transformer block
    # 4. Apply final layer norm
    # 5. Project to vocabulary: x @ wte.T
    pass
```

**Key Insight**: We use `wte.T` (transpose) for the final projection because we're "un-embedding" back to vocabulary space.

**Test it**:
```python
# You'll need to load real weights for this
# But you can test the shape with dummy data
inputs = [1, 2, 3, 4, 5]
vocab_size, n_embd = 50257, 768
# ... create dummy parameters ...
logits = gpt2(inputs, wte, wpe, blocks, ln_f, n_head=12)
print(logits.shape)  # Should be [len(inputs), vocab_size]
```

---

### Phase 6: Generation Loop

#### Step 6.1: Implement `generate()`

**What it does**: Autoregressively generates text token by token

**How it works**:
1. Run model on current tokens
2. Get logits for next token
3. Pick most likely token (greedy sampling)
4. Append to input
5. Repeat

**Your task**:
```python
def generate(inputs, params, n_head, n_tokens_to_generate):
    """
    Autoregressive Generation Loop.
    inputs: list of token IDs (will be modified)
    params: model parameters
    n_head: number of attention heads
    n_tokens_to_generate: how many tokens to generate
    """
    # TODO: Implement this
    # Steps:
    # 1. Make a copy of inputs (don't modify original!)
    # 2. Loop n_tokens_to_generate times:
    #    a. Run gpt2() to get logits
    #    b. Get last token's logits: logits[-1]
    #    c. Pick argmax: np.argmax(next_token_logits)
    #    d. Append to inputs
    # 3. Return final list
    pass
```

**Test it**:
```python
# Load real weights first!
input_ids = tokenizer.encode("Hello")
output_ids = generate(input_ids, params, n_head=12, n_tokens_to_generate=10)
output_text = tokenizer.decode(output_ids)
print(output_text)
```

---

## Testing Your Implementation

### Unit Tests

Run the provided test suite:
```bash
python -m pytest tests/test_pico_gpt.py -v
```

### Manual Testing

Test each function individually:
```python
# Test softmax
x = np.array([1.0, 2.0, 3.0])
result = softmax(x)
assert np.allclose(np.sum(result), 1.0)

# Test layer_norm
x = np.random.randn(5, 10)
g = np.ones(10)
b = np.zeros(10)
result = layer_norm(x, g, b)
assert np.allclose(np.mean(result, axis=-1), 0.0, atol=1e-5)
```

### Integration Test

Run the full model:
```bash
python pico_gpt.py
```

---

## Common Pitfalls & Debugging

### Pitfall 1: Shape Mismatches

**Error**: `ValueError: shapes not aligned`

**Solution**: 
- Print shapes at each step
- Remember: matrix multiplication `A @ B` requires `A.shape[-1] == B.shape[0]`
- Use `.shape` to debug

**Example**:
```python
print(f"x shape: {x.shape}")
print(f"w shape: {w.shape}")
result = x @ w  # Will fail if shapes don't match
```

### Pitfall 2: Forgetting to Expand Dimensions

**Error**: Mask shape doesn't match attention scores

**Solution**: 
- Attention scores: `[n_heads, seq_len, seq_len]`
- Mask: `[seq_len, seq_len]`
- Expand mask: `np.expand_dims(mask, 0)` → `[1, seq_len, seq_len]`

### Pitfall 3: Modifying Input Lists

**Error**: Original input gets modified

**Solution**: Always make a copy:
```python
inputs = list(original_inputs)  # Copy!
```

### Pitfall 4: Numerical Instability

**Error**: NaN or Inf values

**Solution**:
- Use `np.max()` subtraction in softmax
- Add epsilon (`eps`) in layer_norm
- Check for division by zero

### Debugging Tips

1. **Print shapes**: Add `print(f"Shape: {x.shape}")` everywhere
2. **Print values**: Check for NaN/Inf: `print(np.isnan(x).any())`
3. **Test with known values**: Use identity matrices for testing
4. **Start small**: Test with `seq_len=2, n_embd=4` first
5. **Compare with reference**: Use the provided `pico_gpt.py` as reference

---

## Exercises & Challenges

### Beginner Exercises

1. **Implement softmax from scratch** (without looking at the solution)
2. **Visualize attention weights**: Plot which tokens attend to which
3. **Modify the prompt**: Try different prompts and observe outputs
4. **Change generation length**: See how output quality changes

### Intermediate Challenges

1. **Add temperature sampling**: Instead of greedy, use temperature
   ```python
   temperature = 0.8
   probs = softmax(logits / temperature)
   next_token = np.random.choice(vocab_size, p=probs)
   ```

2. **Implement Top-K sampling**: Only sample from top K tokens
3. **Add beam search**: Generate multiple candidates and pick best
4. **Visualize embeddings**: Plot token embeddings in 2D (use PCA)

### Advanced Challenges

1. **Implement nucleus sampling (Top-P)**: Sample from tokens with cumulative probability ≤ p
2. **Add attention visualization**: Show attention patterns for each head
3. **Modify architecture**: Try different numbers of heads or layers
4. **Optimize performance**: Use NumPy optimizations to speed up

---

## Resources

### Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 architecture

### Interactive Resources

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [Transformer from Scratch](https://e2eml.school/transformers.html) - Step-by-step guide

### Code References

- `pico_gpt.py`: Complete reference implementation
- `tests/test_pico_gpt.py`: Test cases to verify your code
- `instructions/student_builder_prompts.md`: Prompts for AI assistants

### Getting Help

1. **Use AI assistants**: Paste your code + prompts from `student_builder_prompts.md`
2. **Check test failures**: Read the error messages carefully
3. **Print intermediate values**: Debug by printing shapes and values
4. **Compare with reference**: Use `pico_gpt.py` as a guide (but try to implement yourself first!)

---

## Progress Checklist

Use this to track your progress:

- [ ] Phase 1: Basic Building Blocks
  - [ ] `softmax()`
  - [ ] `gelu()`
  - [ ] `layer_norm()`
  - [ ] `linear()`
- [ ] Phase 2: Attention Mechanism
  - [ ] `attention()`
  - [ ] `mha()`
- [ ] Phase 3: Feed-Forward Network
  - [ ] `feed_forward()`
- [ ] Phase 4: Transformer Block
  - [ ] `transformer_block()`
- [ ] Phase 5: Full Model
  - [ ] `gpt2()`
- [ ] Phase 6: Generation
  - [ ] `generate()`
- [ ] All tests passing
- [ ] Can generate text successfully
- [ ] Understand how each component works

---

## Final Notes

**Remember**:
- It's okay to struggle! Understanding transformers is hard
- Build incrementally: get one function working before moving to the next
- Test frequently: don't wait until the end to test
- Ask questions: use the prompts in `student_builder_prompts.md`
- Have fun: You're building something amazing!

**Good luck! 🚀**


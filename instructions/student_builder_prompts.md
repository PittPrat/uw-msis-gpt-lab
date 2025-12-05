# Student Builder Prompts

These prompts are designed for students to use with AI coding assistants (ChatGPT, Claude, etc.) to help understand and build the PicoGPT implementation.

## Prompt 1: Understanding the Attention Mechanism

**Use this when you're confused about how attention works:**

```
I am implementing the attention function in PicoGPT. Can you explain step-by-step what happens when we compute:

attention_scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_k)

Specifically:
1. What are the dimensions of q, k, and v before and after this operation?
2. Why do we transpose k, and what does transpose(0, 2, 1) mean?
3. Why do we divide by sqrt(d_k)? What problem does this solve?
4. Use a simple analogy (like a filing system or search engine) to explain the Query-Key-Value concept.
```

## Prompt 2: Understanding Multi-Head Attention

**Use this to understand how multiple attention heads work:**

```
I'm working on the mha (Multi-Head Attention) function. Can you explain:

1. Why do we split the embedding dimension into n_heads * head_dim?
2. What does it mean to have multiple "heads" focusing on different relationships?
3. In a business context (like analyzing customer contracts), what might different heads focus on?
   - Head 1: Grammar and syntax
   - Head 2: ???
   - Head 3: ???
4. How do we merge the heads back together after attention?
```

## Prompt 3: Understanding Layer Normalization

**Use this to understand normalization:**

```
I'm implementing layer_norm. Can you explain:

1. What does it mean to "normalize across the feature dimension"?
2. Why do we need both gamma (g) and beta (b) parameters?
3. What happens if we set gamma=1 and beta=0? What if we don't normalize at all?
4. Why is layer normalization important for deep networks?
```

## Prompt 4: Understanding Residual Connections

**Use this to understand why we add x + ... in transformer blocks:**

```
Looking at the transformer_block function, I see:

    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + feed_forward(layer_norm(x, **ln_2), **mlp)
    
Can you explain:

1. What is a residual connection and why do we use it?
2. What would happen if we removed the "+ x" part?
3. Why is this critical for training deep networks (even though we're only doing inference)?
4. Draw an ASCII diagram showing the data flow through a transformer block.
```

## Prompt 5: Understanding the Full GPT-2 Architecture

**Use this to see the big picture:**

```
I want to understand how the full GPT-2 model works. Can you:

1. Trace the flow of data from input tokens to output logits
2. Explain what happens at each stage:
   - Token embeddings + positional encodings
   - Transformer blocks (12 layers)
   - Final layer normalization
   - Vocabulary projection
3. Why do we use the transpose of the embedding matrix (wte.T) for the final projection?
4. What are "logits" and how do they relate to probabilities?
```

## Prompt 6: Debugging Generation Issues

**Use this if your output is repetitive or nonsensical:**

```
I ran the generate function and got repetitive output like "the the the the". 

1. Looking at the generate function, what creates this behavior?
2. What is "Greedy Sampling" and why might it cause repetition?
3. What is "Nucleus Sampling" (Top-P) and how does it differ?
4. How would I modify the generate function to use Top-P sampling instead?
5. Why do production systems like ChatGPT use Top-P instead of greedy sampling?
```

## Prompt 7: Understanding Matrix Dimensions

**Use this when you're confused about tensor shapes:**

```
I'm getting shape mismatch errors. Can you help me understand the expected dimensions for:

1. Input to gpt2: inputs (list), wte, wpe, blocks
2. After token embedding: what shape is x?
3. In mha: what are the shapes of q, k, v at each step?
4. In attention: what shape should the mask be?
5. Final output: what shape are the logits?

Please provide a dimension trace through one forward pass with seq_len=5, n_embd=768, n_head=12.
```

## Prompt 8: Understanding GELU vs ReLU

**Use this to understand activation functions:**

```
Why does GPT-2 use GELU instead of ReLU?

1. What is the mathematical difference between GELU and ReLU?
2. What does GELU stand for and what does the formula mean?
3. Why might GELU be better for language models?
4. Can you show me what the GELU function looks like graphically compared to ReLU?
```

## Prompt 9: Understanding Causal Masking

**Use this to understand why we prevent "looking ahead":**

```
I see this line in mha:
mask = (1 - np.tri(seq_len)) * -1e10

1. What does np.tri(seq_len) create?
2. Why do we use (1 - tri) instead of just tri?
3. Why multiply by -1e10? Why not -infinity?
4. What would happen if we didn't use a causal mask?
5. Can you visualize what the mask matrix looks like for seq_len=5?
```

## Prompt 10: Understanding the Feed-Forward Network

**Use this to understand the MLP component:**

```
Looking at feed_forward:

1. Why do we expand to 4x the embedding dimension (c_fc)?
2. What does "position-wise" mean in "position-wise feed-forward network"?
3. Why do we use GELU activation between the two linear layers?
4. Why do we contract back to the original dimension (c_proj)?
5. How does this differ from the attention mechanism?
```

## How to Use These Prompts

1. **Copy the prompt** that matches your current confusion
2. **Paste it into ChatGPT/Claude** along with the relevant code section from `pico_gpt.py`
3. **Ask follow-up questions** if something is still unclear
4. **Try implementing** the explanation yourself after understanding it

Remember: The goal is to understand the architecture, not just copy code. Use these prompts to build your understanding step by step!

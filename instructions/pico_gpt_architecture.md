# PicoGPT Architecture Map

This document provides a visual and theoretical map connecting the specific Python functions in `pico_gpt.py` to the architectural concepts in the "Attention Is All You Need" (2017) paper.

> **📊 Interactive Visualization**: Open `architecture_visualization.html` in your browser for a visual, interactive version of this document!

## 1. The Execution Flow (Visual Map)

```mermaid
graph TD
    %% Style Definitions
    classDef input fill:#e0e7ff,stroke:#3730a3,stroke-width:2px;
    classDef norm fill:#f1f5f9,stroke:#475569,stroke-width:2px;
    classDef core fill:#fff7ed,stroke:#c2410c,stroke-width:2px;
    classDef output fill:#dcfce7,stroke:#166534,stroke-width:2px;
    classDef resid fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,stroke-dasharray: 5 5;

    %% Main Flow
    Start(Input Indices<br/>[batch, seq]):::input --> Emb[Embeddings<br/>wte[x] + wpe[i]<br/>Goal: To Vectors]:::input
    
    Emb --> BlockStart
    
    subgraph TransformerBlock ["Loop: Transformer Block (12x)"]
        direction TB
        BlockStart(( )) --> LN1[Layer Norm 1<br/>layer_norm()<br/>Goal: Stability]:::norm
        
        LN1 --> MHA[Multi-Head Attention<br/>mha()<br/>Goal: Parallel Focus]:::core
        
        MHA --> Res1((+)):::resid
        BlockStart --> Res1
        
        Res1 --> LN2[Layer Norm 2<br/>layer_norm()<br/>Goal: Stability]:::norm
        
        LN2 --> FFN[Feed Forward<br/>feed_forward()<br/>Goal: Thinking]:::core
        
        FFN --> Res2((+)):::resid
        Res1 -->|Skip Connection| Res2
    end
    
    Res2 --> LNF[Final Norm<br/>layer_norm()]:::norm
    LNF --> UnEmb[Projection<br/>x @ wte.T<br/>Goal: To Logits]:::output
    UnEmb --> Soft[Softmax<br/>softmax()<br/>Goal: To Probs]:::output
    Soft --> End(Next Token):::output

    %% Annotations for Sub-functions
    MHA -.->|Uses| AttnFunc[Scaled Dot-Product<br/>attention()]:::core
    FFN -.->|Uses| ActFunc[GELU<br/>gelu()]:::core
    FFN -.->|Uses| LinFunc[Linear<br/>linear()]:::core
```

**Legend:**
- 🔵 **Data Prep** (Blue) - Input processing and embeddings
- ⚪ **Normalization** (Gray) - Layer normalization for stability
- 🟠 **Core Intelligence** (Orange) - Attention and feed-forward networks
- 🟢 **Prediction** (Green) - Output generation
- ➕ **Residual Connections** (Green dashed) - Skip connections enabling deep networks

> **Note**: For best visualization, open `architecture_visualization.html` in your browser. The diagram above will render on GitHub and other Markdown viewers that support Mermaid.

---

## 2. Detailed Technical Mapping

Use this table to understand why we write each specific function in the lab.

| Python Function | Technical Concept | The "Why" (Educational Goal) |
|----------------|-------------------|-------------------------------|
| `gelu(x)` | **Non-Linearity** | Teaches that neurons need an "activation threshold" to learn complex patterns. GPT-2 uses GELU (smoother) instead of the standard ReLU. |
| `softmax(x)` | **Probability Distribution** | Converts raw math numbers (logits like 4.5, -1.2) into percentages (probabilities like 90%, 0.1%). Critical for the final prediction. |
| `layer_norm(...)` | **Normalization** | Teaches stability. Without this, the numbers inside the deep network would grow too large (explode) or too small (vanish). |
| `linear(x, w, b)` | **Matrix Transformation** | The fundamental atom of Deep Learning (y = xW + b). Represents a "dense" layer connecting neurons. |
| `attention(...)` | **Scaled Dot-Product Attention** | The Core Concept. Teaches how words "look at" each other. Students implement the famous equation: *Attention(Q, K, V) = softmax(QK<sup>T</sup>/√d<sub>k</sub>)V*. |
| `mha(...)` | **Multi-Head Attention** | Teaches "Parallel Reasoning." One head might focus on grammar, another on dates, another on names. This function splits and recombines them. |
| `feed_forward(...)` | **Position-wise FFN** | The "Brain" of the layer. While Attention gathers context, the FFN processes that context to create meaning. |
| `transformer_block(...)` | **Residual Architecture** | Teaches the "Skip Connection" (Residual). Note the `x = x + ...`. This allows gradients to flow easily, enabling very deep networks (like GPT-4). |
| `gpt2(...)` | **Forward Pass** | The orchestrator. It loops through the blocks 12 times (for GPT-2 Small), showing the sequential nature of deep processing. |
| `generate(...)` | **Autoregression** | The "Chat" loop. Teaches that LLMs generate one word at a time, feeding their own output back in as the next input. |

---

## 3. Data Transformations (Tensor Shapes)

Understanding how data shapes change through the model:

### 1. Input Tokens
- **Shape**: `[Batch, Seq_Len]` (List of Integers)
- **Example**: `inputs = [54, 1200]` represents tokens for "The", "cat"
- **Concept**: Raw token IDs from the tokenizer
- **Code**: `tokenizer.encode(prompt)`

### 2. Embeddings
- **Shape**: `[Batch, Seq_Len, 768]` (Vectors)
- **Concept**: The "High-Dimensional" space where meanings exist. 768 is the hidden size of GPT-2 Small. Each word becomes a vector of 768 floating point numbers.
- **Function**: `x = wte[inputs] + wpe[positions]`
- **Code**: `wte[inputs] + wpe[range(len(inputs))]`

### 3. Attention Matrix
- **Shape**: `[Batch, Heads, Seq_Len, Seq_Len]` (Matrix)
- **Concept**: The "Relevance Map". A square grid showing how much every word attends to every other word. Each cell (i,j) represents how much token i "attends to" token j.
- **Function**: `attention()` computes `Q @ K.T`
- **Code**: `attention_scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_k)`

### 4. Output Logits
- **Shape**: `[Batch, Seq_Len, 50257]` (Vocabulary Size)
- **Concept**: A score for every word in the dictionary (50,257 total) predicting likelihood. The highest score becomes the predicted next token.
- **Function**: `gpt2()` returns logits, `softmax()` converts to probabilities
- **Code**: `logits = x @ wte.T` then `probs = softmax(logits[-1])`

---

## 4. Learning Path

**Recommended Study Order:**

1. **Start with basics**: `linear()`, `softmax()`, `gelu()`
   - These are the building blocks used throughout
   
2. **Understand normalization**: `layer_norm()`
   - Critical for deep network stability
   
3. **Master attention**: `attention()` → `mha()`
   - The core innovation of transformers
   
4. **Build the block**: `feed_forward()` → `transformer_block()`
   - Combines attention and processing
   
5. **Complete the model**: `gpt2()` → `generate()`
   - Full architecture and generation loop

---

## 5. Key Insights

### Why Residual Connections?
The `x = x + ...` pattern allows information to "skip" layers, preventing the vanishing gradient problem. This enables GPT-2 to have 12 layers (and GPT-4 to have many more) without losing information.

### Why Multi-Head Attention?
Instead of one attention mechanism, GPT-2 uses 12 parallel "heads". Each head can focus on different relationships:
- Head 1: Grammar and syntax
- Head 2: Semantic meaning
- Head 3: Temporal relationships
- ... and so on

### Why Layer Normalization?
After each major operation (attention, feed-forward), we normalize the values. This keeps the numbers in a reasonable range and prevents:
- **Exploding gradients**: Values growing too large
- **Vanishing gradients**: Values shrinking to zero

### The 12x Loop
GPT-2 Small processes data through 12 identical transformer blocks. Each block:
1. Gathers context (attention)
2. Processes that context (feed-forward)
3. Passes it to the next block

Think of it like reading a document 12 times, each time understanding it better.

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) - The original Transformer paper
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 architecture details
- `pico_gpt.py` - Complete reference implementation
- `LEARNING_GUIDE.md` - Step-by-step implementation guide

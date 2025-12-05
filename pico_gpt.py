import numpy as np

"""
PicoGPT: A NumPy Implementation of GPT-2
----------------------------------------
This is the core lab file for the "Transformer from Scratch" module.
It implements the GPT-2 architecture using only raw linear algebra.

Educational Goal: Understand the 'Forward Pass' of a Transformer.
Reference: "Attention Is All You Need" (Vaswani et al., 2017)
"""

def gelu(x):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    Used in GPT-2 instead of ReLU.
    Formula: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax(x):
    """
    Standard Softmax function to convert logits into probabilities.
    Input: Vector of numbers (logits).
    Output: Vector of probabilities (0.0 to 1.0) summing to 1.
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps: float = 1e-5):
    """
    Layer Normalization.
    Normalizes the input across the feature dimension.
    x: Input
    g: Gamma (Scale parameter learned by model)
    b: Beta (Shift parameter learned by model)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    """
    A Linear Layer (Dense Layer).
    Equation: y = xW + b
    This is the fundamental building block of Neural Networks.
    """
    return x @ w + b

def attention(q, k, v, mask):
    """
    Scaled Dot-Product Attention (The "Brain" of the Transformer).
    Ref: Equation 1 in Vaswani et al. (2017)
    
    Args:
        q: Query vectors [n_heads, seq_len, head_dim]
        k: Key vectors [n_heads, seq_len, head_dim]
        v: Value vectors [n_heads, seq_len, head_dim]
        mask: Causal mask [seq_len, seq_len] (prevents looking at future tokens)
    
    Returns:
        output: The weighted sum of values [n_heads, seq_len, head_dim]
    """
    # 1. MatMul Q and K
    # Shape: [n_heads, seq_len, seq_len]
    attention_scores = q @ k.transpose(0, 2, 1)
    
    # 2. Scale by sqrt(d_k)
    # This prevents gradients from vanishing (though we are only doing inference here).
    d_k = q.shape[-1]
    attention_scores = attention_scores / np.sqrt(d_k)
    
    # 3. Apply Causal Mask (The "Time Travel" prevention)
    # Sets future positions to -infinity so softmax makes them 0.
    # Expand mask to match attention_scores shape [n_heads, seq_len, seq_len]
    mask_expanded = np.expand_dims(mask, 0)  # [1, seq_len, seq_len]
    attention_scores = attention_scores + mask_expanded
    
    # 4. Softmax (Convert scores to probabilities)
    attention_weights = softmax(attention_scores)
    
    # 5. MatMul with V
    return attention_weights @ v

def mha(x, c_attn, c_proj, n_head):
    """
    Multi-Head Attention (MHA).
    Splits the input into multiple 'heads' so the model can focus on 
    different relationships (e.g., grammar vs. meaning) simultaneously.
    
    Args:
        x: Input tensor [seq_len, n_embd]
        c_attn: Attention projection weights {'w': [n_embd, 3*n_embd], 'b': [3*n_embd]}
        c_proj: Output projection weights {'w': [n_embd, n_embd], 'b': [n_embd]}
        n_head: Number of attention heads
    """
    # Dimensions
    x_shape = x.shape  # [seq_len, n_embd]
    n_embd = x_shape[-1]
    seq_len = x_shape[0]
    
    # 1. Project Input to Q, K, V
    # In GPT-2, Q, K, V are projected in one giant matrix for efficiency.
    x = linear(x, **c_attn)
    
    # 2. Split into Q, K, V
    qkv = np.split(x, 3, axis=-1)
    q, k, v = qkv
    
    # 3. Reshape for Heads
    # Split embedding dim into n_heads * head_dim
    head_dim = n_embd // n_head
    
    # Reshape: [n_head, seq_len, head_dim]
    q = q.reshape(seq_len, n_head, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, n_head, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, n_head, head_dim).transpose(1, 0, 2)
    
    # 4. Create Causal Mask
    # A matrix of 0s and -infs. 
    # (1 - tri) * -1e10 ensures we only look at past tokens.
    mask = (1 - np.tri(seq_len)) * -1e10
    
    # 5. Calculate Attention
    output = attention(q, k, v, mask)
    
    # 6. Merge Heads
    # Reshape back to [seq_len, n_embd]
    output = output.transpose(1, 0, 2).reshape(seq_len, n_embd)
    
    # 7. Final Linear Projection
    output = linear(output, **c_proj)
    
    return output

def feed_forward(x, c_fc, c_proj):
    """
    Position-wise Feed-Forward Network.
    A simple 2-layer neural network applied to every token independently.
    Structure: Input -> Expand (4x) -> GELU -> Contract -> Output
    
    Args:
        x: Input tensor [seq_len, n_embd]
        c_fc: First layer weights {'w': [n_embd, 4*n_embd], 'b': [4*n_embd]}
        c_proj: Second layer weights {'w': [4*n_embd, n_embd], 'b': [n_embd]}
    """
    # Layer 1 (Expansion)
    a = gelu(linear(x, **c_fc))
    # Layer 2 (Contraction)
    return linear(a, **c_proj)

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    """
    A Single Transformer Block.
    Combines Attention and Feed-Forward with Residual Connections.
    Structure:
    x = x + MHA(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
    
    Args:
        x: Input tensor [seq_len, n_embd]
        mlp: Feed-forward network parameters
        attn: Attention parameters
        ln_1: First layer norm parameters {'g': [n_embd], 'b': [n_embd]}
        ln_2: Second layer norm parameters {'g': [n_embd], 'b': [n_embd]}
        n_head: Number of attention heads
    """
    # Attention Sub-Layer with Residual
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    
    # Feed-Forward Sub-Layer with Residual
    x = x + feed_forward(layer_norm(x, **ln_2), **mlp)
    
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    """
    The Full GPT-2 Model Architecture.
    
    Args:
        inputs: List of token IDs (integers)
        wte: Token embedding matrix [vocab_size, n_embd]
        wpe: Positional embedding matrix [max_seq_len, n_embd]
        blocks: List of transformer block parameters
        ln_f: Final layer norm parameters {'g': [n_embd], 'b': [n_embd]}
        n_head: Number of attention heads
    
    Returns:
        logits: Output logits [seq_len, vocab_size]
    """
    # 1. Token Embeddings + Positional Encodings
    # inputs is a list of token IDs (integers)
    # wte[inputs] looks up the vector for each token
    # wpe[range] adds the positional vector (0, 1, 2...)
    x = wte[inputs] + wpe[range(len(inputs))]
    
    # 2. Pass through Transformer Blocks (12 layers in GPT-2 Small)
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    
    # 3. Final Layer Normalization
    x = layer_norm(x, **ln_f)
    
    # 4. Projection to Vocabulary (Un-embedding)
    # Calculates the probability for the NEXT token.
    # Uses the transpose of the embedding matrix.
    return x @ wte.T

def generate(inputs, params, n_head, n_tokens_to_generate):
    """
    Autoregressive Generation Loop.
    This is the "Chat" loop.
    
    Args:
        inputs: List of token IDs (will be modified in place)
        params: Model parameters dictionary
        n_head: Number of attention heads
        n_tokens_to_generate: Number of tokens to generate
    
    Returns:
        List of token IDs including original inputs and generated tokens
    """
    from tqdm import tqdm
    
    # Make a copy to avoid modifying the original list
    inputs = list(inputs)
    
    for _ in tqdm(range(n_tokens_to_generate), "Generating"):
        # 1. Run the model
        logits = gpt2(inputs, **params, n_head=n_head)
        
        # 2. Get the last token's prediction
        next_token_logits = logits[-1]
        
        # 3. Greedy Sampling (Pick the most likely next word)
        next_token_id = np.argmax(next_token_logits)
        
        # 4. Append to input and repeat
        inputs.append(int(next_token_id))
        
    return inputs

if __name__ == "__main__":
    import os
    from transformers import GPT2Tokenizer
    
    # 1. Load Weights (Run setup_weights.py first!)
    if not os.path.exists("gpt2_weights.npz"):
        print("Error: Weights not found! Please run 'python setup_weights.py' first.")
        exit(1)
        
    print("Loading weights...")
    # Allow pickle=True because we saved a dictionary of objects
    params = np.load("gpt2_weights.npz", allow_pickle=True)
    # Re-loading logic:
    # Since np.savez breaks nested dictionaries, we rely on setup_weights saving 'params' as a single object array
    params = params['params'].item() 

    # 2. Setup Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 3. Prompt
    prompt = "Alan Turing theorized that computers would one day become"
    input_ids = tokenizer.encode(prompt)
    
    print(f"\nPrompt: {prompt}")
    print(f"Input IDs: {input_ids}")
    
    # 4. Generate
    # GPT-2 Small has 12 heads
    output_ids = generate(input_ids, params, n_head=12, n_tokens_to_generate=20)
    
    # 5. Decode
    output_text = tokenizer.decode(output_ids)
    print(f"\nGenerated Text:\n{output_text}")


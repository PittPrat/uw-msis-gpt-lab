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

def sample_with_temperature(logits, temperature=1.0, top_k=None, top_p=1.0):
    """
    Sample from logits using temperature scaling with optional Top-K and Top-P filtering.
    
    Args:
        logits: Raw model output [vocab_size]
        temperature: Controls randomness (0.1 = deterministic, 2.0 = very random)
        top_k: If set, only sample from top K tokens (None = no limit)
        top_p: If set, only sample from tokens with cumulative probability <= top_p (1.0 = no limit)
    
    Returns:
        Sampled token ID
    """
    if temperature == 0.0 or temperature < 0.1:
        # Greedy sampling (deterministic)
        return np.argmax(logits)
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Apply Top-K filtering if specified
    if top_k is not None and top_k > 0:
        # Get indices of top-k tokens
        top_k = min(top_k, len(scaled_logits))
        top_k_indices = np.argsort(scaled_logits)[-top_k:]
        # Create a mask: set all other logits to -inf
        mask = np.full_like(scaled_logits, -np.inf)
        mask[top_k_indices] = scaled_logits[top_k_indices]
        scaled_logits = mask
    
    # Convert to probabilities
    probs = softmax(scaled_logits)
    
    # Apply Top-P (nucleus) filtering if specified
    if top_p < 1.0:
        # Sort probabilities in descending order
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        # Calculate cumulative probabilities
        cumsum_probs = np.cumsum(sorted_probs)
        
        # Find tokens to keep (cumulative probability <= top_p)
        keep_mask = cumsum_probs <= top_p
        # Always keep at least the top token
        if not keep_mask[0]:
            keep_mask[0] = True
        
        # Create new probability distribution with only kept tokens
        new_probs = np.zeros_like(probs)
        kept_indices = sorted_indices[keep_mask]
        kept_probs = sorted_probs[keep_mask]
        # Renormalize
        kept_probs = kept_probs / kept_probs.sum()
        new_probs[kept_indices] = kept_probs
        probs = new_probs
    
    # Sample from the filtered distribution
    return np.random.choice(len(probs), p=probs)

def apply_frequency_penalty(logits, generated_tokens, penalty=0.0):
    """
    Apply frequency penalty to reduce repetition.
    
    Args:
        logits: Raw model output [vocab_size]
        generated_tokens: List of previously generated token IDs
        penalty: Penalty strength (0.0 = no penalty, 2.0 = strong penalty)
    
    Returns:
        Adjusted logits
    """
    if penalty == 0.0 or len(generated_tokens) == 0:
        return logits
    
    # Count frequency of each token in generated sequence
    token_counts = {}
    for token_id in generated_tokens:
        token_counts[token_id] = token_counts.get(token_id, 0) + 1
    
    # Apply penalty: subtract penalty * count for each token
    adjusted_logits = logits.copy()
    for token_id, count in token_counts.items():
        adjusted_logits[token_id] -= penalty * count
    
    return adjusted_logits

def generate(inputs, params, n_head, n_tokens_to_generate, temperature=1.0, frequency_penalty=0.0, top_k=50, top_p=0.9):
    """
    Autoregressive Generation Loop.
    This is the "Chat" loop.
    
    Args:
        inputs: List of token IDs (will be modified in place)
        params: Model parameters dictionary
        n_head: Number of attention heads
        n_tokens_to_generate: Number of tokens to generate
        temperature: Sampling temperature (0.1-2.0). Lower = deterministic, Higher = creative. Default: 1.0
        frequency_penalty: Penalty for repetition (0.0-2.0). Higher = less repetition. Default: 0.0
        top_k: Sample only from top K tokens (None = no limit). Default: 50 (recommended for quality)
        top_p: Nucleus sampling - sample from tokens with cumulative prob <= top_p. Default: 0.9 (recommended)
    
    Returns:
        List of token IDs including original inputs and generated tokens
    """
    from tqdm import tqdm
    
    # Make a copy to avoid modifying the original list
    inputs = list(inputs)
    generated_tokens = []  # Track generated tokens for frequency penalty
    
    for _ in tqdm(range(n_tokens_to_generate), "Generating"):
        # 1. Run the model
        logits = gpt2(inputs, **params, n_head=n_head)
        
        # 2. Get the last token's prediction
        next_token_logits = logits[-1]
        
        # 3. Apply frequency penalty if enabled
        if frequency_penalty > 0.0:
            next_token_logits = apply_frequency_penalty(
                next_token_logits, 
                generated_tokens, 
                penalty=frequency_penalty
            )
        
        # 4. Sample with temperature and filtering
        if temperature == 1.0 and top_k is None and top_p >= 1.0:
            # Pure greedy sampling (original behavior)
            next_token_id = np.argmax(next_token_logits)
        else:
            # Temperature sampling with Top-K/Top-P for better quality
            next_token_id = sample_with_temperature(
                next_token_logits, 
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # 5. Append to input and repeat
        inputs.append(int(next_token_id))
        generated_tokens.append(int(next_token_id))
        
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
    # Using quality defaults: top_k=50, top_p=0.9 for better output quality
    output_ids = generate(
        input_ids, 
        params, 
        n_head=12, 
        n_tokens_to_generate=20,
        temperature=0.8,  # Slightly lower for more focused output
        top_k=50,         # Sample from top 50 tokens
        top_p=0.9,        # Nucleus sampling
        frequency_penalty=0.3  # Reduce repetition
    )
    
    # 5. Decode
    output_text = tokenizer.decode(output_ids)
    print(f"\nGenerated Text:\n{output_text}")




# Guide: Running PicoGPT in Google Colab

Since Colab runs in the cloud, we need a special "Tunnel" to view the Streamlit UI.
Below are the 4 Code Cells you need to put in your Colab Notebook.

---

## Cell 1: Install & Setup (Run Once)

This installs the libraries and downloads the GPT-2 weights.

```python
# 1. Install Dependencies
!pip install -q numpy torch transformers tokenizers streamlit tqdm

# 2. Download & Convert Weights (Setup Script)
import os
import numpy as np
import torch
from transformers import GPT2Model

print("Downloading GPT-2 weights from Hugging Face...")
hf_model = GPT2Model.from_pretrained("gpt2")
state_dict = hf_model.state_dict()

params = {}

print("Converting weights to NumPy format...")
for key, value in state_dict.items():
    np_arr = value.detach().cpu().numpy()
    params[key] = np_arr

# Clean and map parameters
clean_params = {
    "wte": params["wte.weight"],
    "wpe": params["wpe.weight"],
    "blocks": []
}

for i in range(12):
    prefix = f"h.{i}."
    block_params = {
        "ln_1": {
            "g": params[f"{prefix}ln_1.weight"],
            "b": params[f"{prefix}ln_1.bias"]
        },
        "attn": {
            "c_attn": {
                "w": params[f"{prefix}attn.c_attn.weight"],
                "b": params[f"{prefix}attn.c_attn.bias"]
            },
            "c_proj": {
                "w": params[f"{prefix}attn.c_proj.weight"],
                "b": params[f"{prefix}attn.c_proj.bias"]
            }
        },
        "ln_2": {
            "g": params[f"{prefix}ln_2.weight"],
            "b": params[f"{prefix}ln_2.bias"]
        },
        "mlp": {
            "c_fc": {
                "w": params[f"{prefix}mlp.c_fc.weight"],
                "b": params[f"{prefix}mlp.c_fc.bias"]
            },
            "c_proj": {
                "w": params[f"{prefix}mlp.c_proj.weight"],
                "b": params[f"{prefix}mlp.c_proj.bias"]
            }
        }
    }
    clean_params["blocks"].append(block_params)

clean_params["ln_f"] = {
    "g": params["ln_f.weight"],
    "b": params["ln_f.bias"]
}

# Save weights (wrapped in 'params' key for proper loading)
np.savez("gpt2_weights.npz", params=clean_params)
print("✅ Setup Complete! Weights saved to gpt2_weights.npz")
```

---

## Cell 2: The Implementation (The Work Area)

**Crucial Step**: We use `%%writefile` to save this cell as a Python file so the UI can import it.

```python
%%writefile pico_gpt.py
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
    """
    # Attention Sub-Layer with Residual
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    
    # Feed-Forward Sub-Layer with Residual
    x = x + feed_forward(layer_norm(x, **ln_2), **mlp)
    
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    """
    The Full GPT-2 Model Architecture.
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
```

---

## Cell 3: The UI Code

This writes the UI file to the Colab disk.

```python
%%writefile web_ui.py
"""
Streamlit UI for PicoGPT
------------------------
This file wraps the raw NumPy implementation in a web interface.
Run this using: streamlit run web_ui.py
"""

import streamlit as st
import numpy as np
import os
from transformers import GPT2Tokenizer
import pico_gpt as model  # Import the PicoGPT implementation

# Page Config - MUST be first Streamlit command
st.set_page_config(page_title="MSIS PicoGPT", page_icon="🤖")

st.title("🤖 PicoGPT: The NumPy Transformer")
st.markdown("""
This is a **educational implementation** of GPT-2 running on raw NumPy. 
There is no PyTorch 'magic' happening behind the scenes during inference—just matrix multiplication!
""")

# Sidebar for controls
with st.sidebar:
    st.header("Model Controls")
    n_tokens = st.slider("Tokens to Generate", min_value=1, max_value=50, value=20)
    
    st.divider()
    
    st.subheader("Sampling Parameters")
    
    # Quality presets
    quality_mode = st.selectbox(
        "Quality Mode",
        ["High Quality (Recommended)", "Balanced", "Creative", "Custom"],
        help="Presets optimize Top-K and Top-P for best results"
    )
    
    # Set defaults based on mode
    if quality_mode == "High Quality (Recommended)":
        default_temp, default_top_k, default_top_p, default_freq = 0.8, 50, 0.9, 0.3
    elif quality_mode == "Balanced":
        default_temp, default_top_k, default_top_p, default_freq = 1.0, 40, 0.95, 0.2
    elif quality_mode == "Creative":
        default_temp, default_top_k, default_top_p, default_freq = 1.2, 30, 0.85, 0.4
    else:  # Custom
        default_temp, default_top_k, default_top_p, default_freq = 1.0, 50, 0.9, 0.0
    
    temperature = st.slider(
        "Temperature", 
        min_value=0.1, 
        max_value=2.0, 
        value=default_temp, 
        step=0.1,
        help="Controls randomness: Lower = more deterministic, Higher = more creative"
    )
    
    top_k = st.slider(
        "Top-K", 
        min_value=1, 
        max_value=100, 
        value=default_top_k, 
        step=1,
        help="Sample only from top K most likely tokens. Higher = more diverse, Lower = more focused. Recommended: 40-50"
    )
    
    top_p = st.slider(
        "Top-P (Nucleus)", 
        min_value=0.1, 
        max_value=1.0, 
        value=default_top_p, 
        step=0.05,
        help="Sample from tokens with cumulative probability ≤ P. Higher = more diverse. Recommended: 0.9"
    )
    
    frequency_penalty = st.slider(
        "Frequency Penalty", 
        min_value=0.0, 
        max_value=2.0, 
        value=default_freq, 
        step=0.1,
        help="Penalizes repeated tokens: Higher = less repetition. Recommended: 0.2-0.4"
    )
    
    st.caption("💡 **Tip**: Top-K and Top-P work together to improve quality by filtering out low-probability tokens.")
    
    st.divider()
    st.info("Note: Since this runs on CPU with raw NumPy, generation might be slow (approx 1 token/sec). This is expected!")

# 1. Load Resources (Cached so it doesn't reload on every click)
@st.cache_resource
def load_resources():
    # Load params
    params = np.load("gpt2_weights.npz", allow_pickle=True)
    params = params['params'].item()
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    return params, tokenizer

# Check for weights file first (before calling cached function)
weights_exist = os.path.exists("gpt2_weights.npz")

if weights_exist:
    with st.spinner("Loading Weights & Tokenizer..."):
        params, tokenizer = load_resources()
else:
    st.error("⚠️ Weights file not found! Please run `python setup_weights.py` first.")
    params, tokenizer = None, None

# Use sampling functions from pico_gpt module
sample_with_temperature = model.sample_with_temperature
apply_frequency_penalty = model.apply_frequency_penalty

# 2. User Input
prompt = st.text_area("Enter your prompt:", value="Alan Turing theorized that computers would one day become")

# 3. Generation Logic
if st.button("Generate Text", type="primary"):
    if not params:
        st.stop()
        
    # Container for the output
    output_container = st.empty()
    
    # Encode
    input_ids = tokenizer.encode(prompt)
    
    # We'll use a modified generation loop here to update the UI in real-time
    # Copying the logic from pico_gpt.generate but adding UI updates
    current_ids = list(input_ids)
    generated_tokens = []  # Track generated tokens for frequency penalty
    
    # Display initial prompt
    output_text = prompt
    output_container.markdown(f"**Output:**\n\n{output_text}")
    
    progress_bar = st.progress(0)
    
    # Display current settings
    with st.expander("⚙️ Generation Settings", expanded=False):
        st.write(f"**Mode:** {quality_mode}")
        st.write(f"**Temperature:** {temperature} {'(Deterministic)' if temperature < 0.5 else '(Creative)' if temperature > 1.0 else '(Balanced)'}")
        st.write(f"**Top-K:** {top_k} {'(All tokens)' if top_k >= 50257 else f'(Top {top_k} tokens)'}")
        st.write(f"**Top-P:** {top_p} {'(All tokens)' if top_p >= 1.0 else f'(Nucleus sampling)'}")
        st.write(f"**Frequency Penalty:** {frequency_penalty} {'(No penalty)' if frequency_penalty == 0.0 else '(Reducing repetition)'}")
    
    for i in range(n_tokens):
        # Update progress
        progress_bar.progress((i + 1) / n_tokens)
        
        # Run model (Imported from pico_gpt)
        logits = model.gpt2(current_ids, **params, n_head=12)
        next_token_logits = logits[-1]
        
        # Apply frequency penalty
        if frequency_penalty > 0.0:
            next_token_logits = apply_frequency_penalty(
                next_token_logits, 
                generated_tokens, 
                penalty=frequency_penalty
            )
        
        # Sample with temperature, Top-K, and Top-P for better quality
        # Only apply top_k if it's less than vocabulary size
        effective_top_k = top_k if top_k < len(next_token_logits) else None
        
        next_token_id = sample_with_temperature(
            next_token_logits, 
            temperature=temperature,
            top_k=effective_top_k,
            top_p=top_p
        )
        
        # Append
        current_ids.append(int(next_token_id))
        generated_tokens.append(int(next_token_id))
        
        # Decode and update UI
        new_word = tokenizer.decode([next_token_id])
        output_text += new_word
        output_container.markdown(f"**Output:**\n\n{output_text}")

    st.success("Generation Complete!")
    
    # Analysis Section (Optional educational add-on)
    with st.expander("See Under the Hood (Last Step Logits)"):
        # Show the top 5 candidates for the very last token generated
        # Use the original logits before temperature/frequency penalty for display
        final_logits = model.gpt2(current_ids, **params, n_head=12)
        probs = model.softmax(final_logits[-1] / temperature if temperature > 0 else final_logits[-1])
        top_k_indices = np.argsort(probs)[-5:][::-1]
        
        st.write("Top 5 Predictions for the last word:")
        for idx in top_k_indices:
            word = tokenizer.decode([idx])
            prob = probs[idx]
            st.write(f"- **'{word}'**: {prob:.2%}")
```

---

## Cell 4: The Magic Launch (Tunneling)

This runs the app and gives you a clickable URL.

**Option 1: Using localtunnel (Recommended)**

```python
# Install localtunnel
!npm install -g localtunnel

# Run Streamlit in the background
import subprocess
import time

# Start Streamlit
process = subprocess.Popen(
    ["streamlit", "run", "web_ui.py", "--server.headless", "true", "--server.port", "8501"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait a moment for Streamlit to start
time.sleep(3)

# Create tunnel
!npx localtunnel --port 8501
```

**Option 2: Using pyngrok (Alternative)**

```python
# Install pyngrok
!pip install -q pyngrok

# Run Streamlit in the background
import subprocess
import time
from pyngrok import ngrok

# Start Streamlit
process = subprocess.Popen(
    ["streamlit", "run", "web_ui.py", "--server.headless", "true", "--server.port", "8501"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for Streamlit to start
time.sleep(3)

# Create tunnel
public_url = ngrok.connect(8501)
print(f"🌐 Your app is live at: {public_url}")
```

---

## 🛑 Important Notes

### When Using localtunnel:

1. **After running Cell 4**, you'll see output like:
   ```
   your url is: https://funny-cat-55.loca.lt
   ```

2. **Click that link** to open the web UI

3. **If asked for a Tunnel Password**:
   - Run this in a new cell: `!curl ipv4.icanhazip.com`
   - Copy the IP address shown
   - Paste it into the tunnel website

### When Using pyngrok:

1. **After running Cell 4**, you'll see a URL printed
2. **Click the URL** to open the web UI
3. **No password needed** (but may require ngrok account for free tier)

### Troubleshooting:

- **"Module not found"**: Make sure you ran Cell 2 (`%%writefile pico_gpt.py`)
- **"Weights not found"**: Make sure you ran Cell 1 completely
- **"Port already in use"**: Restart the Colab runtime (Runtime → Restart runtime)
- **Tunnel not working**: Try the alternative tunneling method

### Performance Notes:

- Generation is **slow** (~1 token/second) because it's pure NumPy on CPU
- This is **expected** and educational - you're seeing the raw computation!
- For faster generation, consider using GPU runtime (but still NumPy-based)

---

## Quick Reference

**Cell Order:**
1. Cell 1: Install & Setup (one-time)
2. Cell 2: Implementation Code (`pico_gpt.py`)
3. Cell 3: UI Code (`web_ui.py`)
4. Cell 4: Launch & Tunnel

**Files Created:**
- `gpt2_weights.npz` (475MB) - Model weights
- `pico_gpt.py` - PicoGPT implementation
- `web_ui.py` - Streamlit interface

**Expected Runtime:**
- Cell 1: ~2-3 minutes (downloads weights)
- Cell 2: Instant (writes file)
- Cell 3: Instant (writes file)
- Cell 4: ~10 seconds (starts server + tunnel)

---

Good luck! 🚀

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


import streamlit as st
import numpy as np
import os
from transformers import GPT2Tokenizer
import pico_gpt_lab as model  # Import the student's lab code

"""
Streamlit UI for PicoGPT
------------------------
This file wraps the raw NumPy implementation in a web interface.
Run this using: streamlit run app_ui.py
"""

# Page Config
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
    st.info("Note: Since this runs on CPU with raw NumPy, generation might be slow (approx 1 token/sec). This is expected!")

# 1. Load Resources (Cached so it doesn't reload on every click)
@st.cache_resource
def load_resources():
    # Check for weights
    if not os.path.exists("gpt2_weights.npz"):
        st.error("⚠️ Weights file not found! Please run `setup_weights.py` first.")
        return None, None
    
    with st.spinner("Loading Weights & Tokenizer..."):
        # Load params
        params = np.load("gpt2_weights.npz", allow_pickle=True)
        params = params['params'].item()
        
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
    return params, tokenizer

params, tokenizer = load_resources()

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
    # Copying the logic from pico_gpt_lab.generate but adding UI updates
    current_ids = list(input_ids)
    
    # Display initial prompt
    output_text = prompt
    output_container.markdown(f"**Output:**\n\n{output_text}")
    
    progress_bar = st.progress(0)
    
    for i in range(n_tokens):
        # Update progress
        progress_bar.progress((i + 1) / n_tokens)
        
        # Run model (Imported from lab)
        logits = model.gpt2(current_ids, **params, n_head=12)
        next_token_logits = logits[-1]
        next_token_id = np.argmax(next_token_logits)
        
        # Append
        current_ids.append(next_token_id)
        
        # Decode and update UI
        new_word = tokenizer.decode([next_token_id])
        output_text += new_word
        output_container.markdown(f"**Output:**\n\n{output_text}")

    st.success("Generation Complete!")
    
    # Analysis Section (Optional educational add-on)
    with st.expander("See Under the Hood (Last Step Logits)"):
        # Show the top 5 candidates for the very last token generated
        probs = model.softmax(next_token_logits)
        top_k_indices = np.argsort(probs)[-5:][::-1]
        
        st.write("Top 5 Predictions for the last word:")
        for idx in top_k_indices:
            word = tokenizer.decode([idx])
            prob = probs[idx]
            st.write(f"- **'{word}'**: {prob:.2%}")
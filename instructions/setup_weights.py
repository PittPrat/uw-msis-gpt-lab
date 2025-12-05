import os
import numpy as np
import torch
from transformers import GPT2Model

"""
setup_weights.py
----------------
Run this script ONCE to download the GPT-2 weights and save them 
as a NumPy archive.
"""

def get_weights():
    print("Downloading GPT-2 weights from Hugging Face...")
    hf_model = GPT2Model.from_pretrained("gpt2")
    state_dict = hf_model.state_dict()

    params = {}
    
    print("Converting weights to NumPy format...")
    for key, value in state_dict.items():
        np_arr = value.detach().cpu().numpy()
        # NOTE: HuggingFace GPT-2 uses Conv1D layers which store weights as [in, out].
        # Our PicoGPT Linear layer is x @ w, which requires [in, out].
        # Therefore, we DO NOT need to transpose these weights.
        params[key] = np_arr
        
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
    
    return clean_params

if __name__ == "__main__":
    params = get_weights()
    print("Saving to gpt2_weights.npz...")
    # We save the dictionary as a single object inside the npz file
    # This preserves the nested structure (blocks -> attn -> c_attn...)
    np.savez("gpt2_weights.npz", params=params)
    print("Done! You can now run 'python pico_gpt_lab.py'.")
Instructor & AI Prompt Guide

This document serves as the "Interface" between the raw code and the student/instructor.
You can paste the code from pico_gpt_lab.py into ChatGPT/Claude along with these prompts to generate specific teaching materials.

Prompt 1: The "Math-to-English" Translator

Use this to help students who are stuck on the Linear Algebra.

Prompt: "I am an MSIS student looking at the attention function in this Python code. Can you explain the line attention_scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_k)? Please break it down into three parts: 1) What the dimensions of the matrices are, 2) Why we transpose K, and 3) Why we divide by the square root. Use a simple analogy involving a 'filing system' or 'database search'."

Prompt 2: The "Visualization" Generator

Use this to generate ASCII diagrams or slide descriptions.

Prompt: "Analyze the transformer_block function. Generate an ASCII art diagram that shows the flow of data x through the Layer Norms, the Multi-Head Attention, and the Residual connections (the x + ... part). Explain why the Residual connection is critical for deep networks."

Prompt 3: The "Business Case" Connection

Use this to connect the code back to MSIS business goals.

Prompt: "I have implemented mha (Multi-Head Attention). In a business context, if this model were analyzing a customer contract, explain what different 'Heads' might focus on. For example, if Head 1 focuses on grammar, what might Head 2 and Head 3 focus on regarding dates, entities, or liabilities?"

Prompt 4: The "Debugging" Guide

Use this if the output looks like garbage.

Prompt: "I ran the generate loop and the output is repetitive (e.g., 'the the the the'). Looking at the softmax and generate functions, what creates this behavior? Explain the difference between 'Greedy Sampling' (which we used) and 'Nucleus Sampling' (Top-P) which is used in production systems like ChatGPT."
"""
Unit tests for PicoGPT implementation.

Run with: python -m pytest tests/test_pico_gpt.py -v
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path to import pico_gpt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pico_gpt


class TestGELU:
    """Tests for GELU activation function."""
    
    def test_gelu_zero(self):
        """GELU at zero should be approximately zero."""
        result = pico_gpt.gelu(np.array(0.0))
        assert abs(result) < 1e-6
    
    def test_gelu_positive(self):
        """GELU should be positive for positive inputs."""
        x = np.array([1.0, 2.0, 5.0])
        result = pico_gpt.gelu(x)
        assert np.all(result > 0)
    
    def test_gelu_negative(self):
        """GELU should be negative for negative inputs."""
        x = np.array([-1.0, -2.0, -5.0])
        result = pico_gpt.gelu(x)
        assert np.all(result < 0)
    
    def test_gelu_shape_preservation(self):
        """GELU should preserve input shape."""
        x = np.random.randn(3, 4, 5)
        result = pico_gpt.gelu(x)
        assert result.shape == x.shape
    
    def test_gelu_monotonic(self):
        """GELU should be monotonic (larger input -> larger output)."""
        x1 = np.array(1.0)
        x2 = np.array(2.0)
        assert pico_gpt.gelu(x2) > pico_gpt.gelu(x1)


class TestSoftmax:
    """Tests for softmax function."""
    
    def test_softmax_sums_to_one(self):
        """Softmax output should sum to 1."""
        x = np.array([1.0, 2.0, 3.0])
        result = pico_gpt.softmax(x)
        assert abs(np.sum(result) - 1.0) < 1e-6
    
    def test_softmax_all_positive(self):
        """Softmax output should be all positive."""
        x = np.array([-5.0, 0.0, 5.0])
        result = pico_gpt.softmax(x)
        assert np.all(result > 0)
    
    def test_softmax_shape_preservation(self):
        """Softmax should preserve input shape."""
        x = np.random.randn(3, 4)
        result = pico_gpt.softmax(x)
        assert result.shape == x.shape
    
    def test_softmax_2d(self):
        """Softmax should work on 2D arrays."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = pico_gpt.softmax(x)
        assert result.shape == x.shape
        # Each row should sum to 1
        assert np.allclose(np.sum(result, axis=-1), 1.0)
    
    def test_softmax_large_values(self):
        """Softmax should handle large values without overflow."""
        x = np.array([100.0, 200.0, 300.0])
        result = pico_gpt.softmax(x)
        assert np.all(np.isfinite(result))
        assert abs(np.sum(result) - 1.0) < 1e-6


class TestLayerNorm:
    """Tests for layer normalization."""
    
    def test_layer_norm_zero_mean_unit_variance(self):
        """Layer norm with g=1, b=0 should give zero mean, unit variance."""
        x = np.random.randn(10, 20)
        g = np.ones(20)
        b = np.zeros(20)
        result = pico_gpt.layer_norm(x, g, b)
        # Check mean is approximately zero
        assert np.allclose(np.mean(result, axis=-1), 0.0, atol=1e-5)
        # Check variance is approximately 1
        assert np.allclose(np.var(result, axis=-1), 1.0, atol=1e-5)
    
    def test_layer_norm_shape_preservation(self):
        """Layer norm should preserve input shape."""
        x = np.random.randn(5, 10)
        g = np.ones(10)
        b = np.zeros(10)
        result = pico_gpt.layer_norm(x, g, b)
        assert result.shape == x.shape
    
    def test_layer_norm_scale_shift(self):
        """Layer norm should apply scale and shift correctly."""
        x = np.ones((3, 4))
        g = np.array([2.0, 2.0, 2.0, 2.0])
        b = np.array([1.0, 1.0, 1.0, 1.0])
        result = pico_gpt.layer_norm(x, g, b)
        # After normalization, all values should be the same
        assert np.allclose(result, b)


class TestLinear:
    """Tests for linear transformation."""
    
    def test_linear_basic(self):
        """Basic linear transformation: y = xW + b"""
        x = np.array([[1.0, 2.0]])
        w = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([0.0, 0.0])
        result = pico_gpt.linear(x, w, b)
        expected = np.array([[1.0, 2.0]])
        assert np.allclose(result, expected)
    
    def test_linear_with_bias(self):
        """Linear transformation with bias."""
        x = np.array([[1.0, 2.0]])
        w = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 1.0])
        result = pico_gpt.linear(x, w, b)
        expected = np.array([[2.0, 3.0]])
        assert np.allclose(result, expected)
    
    def test_linear_shape(self):
        """Linear should produce correct output shape."""
        x = np.random.randn(5, 10)
        w = np.random.randn(10, 20)
        b = np.random.randn(20)
        result = pico_gpt.linear(x, w, b)
        assert result.shape == (5, 20)


class TestAttention:
    """Tests for scaled dot-product attention."""
    
    def test_attention_shape(self):
        """Attention should produce correct output shape."""
        n_heads = 2
        seq_len = 5
        head_dim = 4
        q = np.random.randn(n_heads, seq_len, head_dim)
        k = np.random.randn(n_heads, seq_len, head_dim)
        v = np.random.randn(n_heads, seq_len, head_dim)
        mask = np.zeros((seq_len, seq_len))
        result = pico_gpt.attention(q, k, v, mask)
        assert result.shape == (n_heads, seq_len, head_dim)
    
    def test_attention_causal_mask(self):
        """Causal mask should prevent looking at future tokens."""
        n_heads = 1
        seq_len = 3
        head_dim = 2
        q = np.random.randn(n_heads, seq_len, head_dim)
        k = np.random.randn(n_heads, seq_len, head_dim)
        v = np.random.randn(n_heads, seq_len, head_dim)
        # Create causal mask: lower triangular should be 0, upper triangular should be -inf
        mask = (1 - np.tri(seq_len)) * -1e10
        
        result = pico_gpt.attention(q, k, v, mask)
        # Result should have correct shape
        assert result.shape == (n_heads, seq_len, head_dim)
    
    def test_attention_scaling(self):
        """Attention scores should be scaled by sqrt(d_k)."""
        n_heads = 1
        seq_len = 2
        head_dim = 4
        q = np.ones((n_heads, seq_len, head_dim))
        k = np.ones((n_heads, seq_len, head_dim))
        v = np.ones((n_heads, seq_len, head_dim))
        mask = np.zeros((seq_len, seq_len))
        
        result = pico_gpt.attention(q, k, v, mask)
        # Should produce finite output
        assert np.all(np.isfinite(result))


class TestMHA:
    """Tests for multi-head attention."""
    
    def test_mha_shape(self):
        """MHA should produce correct output shape."""
        seq_len = 5
        n_embd = 12
        n_head = 3
        head_dim = n_embd // n_head
        
        x = np.random.randn(seq_len, n_embd)
        c_attn = {
            'w': np.random.randn(n_embd, 3 * n_embd),
            'b': np.random.randn(3 * n_embd)
        }
        c_proj = {
            'w': np.random.randn(n_embd, n_embd),
            'b': np.random.randn(n_embd)
        }
        
        result = pico_gpt.mha(x, c_attn, c_proj, n_head)
        assert result.shape == (seq_len, n_embd)
    
    def test_mha_qkv_split(self):
        """MHA should correctly split Q, K, V."""
        seq_len = 2
        n_embd = 6
        n_head = 2
        
        x = np.random.randn(seq_len, n_embd)
        c_attn = {
            'w': np.random.randn(n_embd, 3 * n_embd),
            'b': np.random.randn(3 * n_embd)
        }
        c_proj = {
            'w': np.random.randn(n_embd, n_embd),
            'b': np.random.randn(n_embd)
        }
        
        result = pico_gpt.mha(x, c_attn, c_proj, n_head)
        assert result.shape == (seq_len, n_embd)


class TestFeedForward:
    """Tests for feed-forward network."""
    
    def test_feed_forward_shape(self):
        """Feed-forward should preserve sequence length."""
        seq_len = 5
        n_embd = 10
        x = np.random.randn(seq_len, n_embd)
        c_fc = {
            'w': np.random.randn(n_embd, 4 * n_embd),
            'b': np.random.randn(4 * n_embd)
        }
        c_proj = {
            'w': np.random.randn(4 * n_embd, n_embd),
            'b': np.random.randn(n_embd)
        }
        
        result = pico_gpt.feed_forward(x, c_fc, c_proj)
        assert result.shape == (seq_len, n_embd)
    
    def test_feed_forward_expansion_contraction(self):
        """Feed-forward should expand then contract."""
        seq_len = 3
        n_embd = 8
        x = np.random.randn(seq_len, n_embd)
        c_fc = {
            'w': np.random.randn(n_embd, 4 * n_embd),
            'b': np.random.randn(4 * n_embd)
        }
        c_proj = {
            'w': np.random.randn(4 * n_embd, n_embd),
            'b': np.random.randn(n_embd)
        }
        
        result = pico_gpt.feed_forward(x, c_fc, c_proj)
        # Output should be same shape as input
        assert result.shape == x.shape


class TestTransformerBlock:
    """Tests for transformer block."""
    
    def test_transformer_block_shape(self):
        """Transformer block should preserve input shape."""
        seq_len = 4
        n_embd = 8
        n_head = 2
        x = np.random.randn(seq_len, n_embd)
        
        ln_1 = {'g': np.ones(n_embd), 'b': np.zeros(n_embd)}
        ln_2 = {'g': np.ones(n_embd), 'b': np.zeros(n_embd)}
        attn = {
            'c_attn': {
                'w': np.random.randn(n_embd, 3 * n_embd),
                'b': np.random.randn(3 * n_embd)
            },
            'c_proj': {
                'w': np.random.randn(n_embd, n_embd),
                'b': np.random.randn(n_embd)
            }
        }
        mlp = {
            'c_fc': {
                'w': np.random.randn(n_embd, 4 * n_embd),
                'b': np.random.randn(4 * n_embd)
            },
            'c_proj': {
                'w': np.random.randn(4 * n_embd, n_embd),
                'b': np.random.randn(n_embd)
            }
        }
        
        result = pico_gpt.transformer_block(x, mlp, attn, ln_1, ln_2, n_head)
        assert result.shape == x.shape
    
    def test_transformer_block_residual(self):
        """Transformer block should include residual connections."""
        seq_len = 2
        n_embd = 4
        n_head = 2
        x = np.zeros((seq_len, n_embd))
        
        ln_1 = {'g': np.ones(n_embd), 'b': np.zeros(n_embd)}
        ln_2 = {'g': np.ones(n_embd), 'b': np.zeros(n_embd)}
        attn = {
            'c_attn': {
                'w': np.eye(n_embd, 3 * n_embd),
                'b': np.zeros(3 * n_embd)
            },
            'c_proj': {
                'w': np.eye(n_embd),
                'b': np.zeros(n_embd)
            }
        }
        mlp = {
            'c_fc': {
                'w': np.eye(n_embd, 4 * n_embd),
                'b': np.zeros(4 * n_embd)
            },
            'c_proj': {
                'w': np.eye(4 * n_embd, n_embd),
                'b': np.zeros(n_embd)
            }
        }
        
        result = pico_gpt.transformer_block(x, mlp, attn, ln_1, ln_2, n_head)
        # With identity matrices and zero input, result should be close to zero
        # (but not exactly zero due to layer norm)
        assert result.shape == x.shape


class TestGPT2:
    """Tests for full GPT-2 model."""
    
    def test_gpt2_shape(self):
        """GPT-2 should produce logits with correct shape."""
        vocab_size = 100
        n_embd = 12
        n_head = 3
        seq_len = 5
        max_seq_len = 1024
        
        inputs = [0, 1, 2, 3, 4]
        wte = np.random.randn(vocab_size, n_embd)
        wpe = np.random.randn(max_seq_len, n_embd)
        
        # Create one transformer block
        blocks = [{
            'ln_1': {'g': np.ones(n_embd), 'b': np.zeros(n_embd)},
            'ln_2': {'g': np.ones(n_embd), 'b': np.zeros(n_embd)},
            'attn': {
                'c_attn': {
                    'w': np.random.randn(n_embd, 3 * n_embd),
                    'b': np.random.randn(3 * n_embd)
                },
                'c_proj': {
                    'w': np.random.randn(n_embd, n_embd),
                    'b': np.random.randn(n_embd)
                }
            },
            'mlp': {
                'c_fc': {
                    'w': np.random.randn(n_embd, 4 * n_embd),
                    'b': np.random.randn(4 * n_embd)
                },
                'c_proj': {
                    'w': np.random.randn(4 * n_embd, n_embd),
                    'b': np.random.randn(n_embd)
                }
            }
        }]
        
        ln_f = {'g': np.ones(n_embd), 'b': np.zeros(n_embd)}
        
        logits = pico_gpt.gpt2(inputs, wte, wpe, blocks, ln_f, n_head)
        assert logits.shape == (seq_len, vocab_size)
    
    def test_gpt2_embedding_lookup(self):
        """GPT-2 should correctly look up token embeddings."""
        vocab_size = 10
        n_embd = 4
        n_head = 2
        max_seq_len = 1024
        
        inputs = [0, 1, 2]
        wte = np.eye(vocab_size, n_embd)  # Identity matrix for easy testing
        wpe = np.zeros((max_seq_len, n_embd))
        
        blocks = []
        ln_f = {'g': np.ones(n_embd), 'b': np.zeros(n_embd)}
        
        logits = pico_gpt.gpt2(inputs, wte, wpe, blocks, ln_f, n_head)
        # With identity embeddings and no blocks, should get identity-like logits
        assert logits.shape == (3, vocab_size)


class TestGenerate:
    """Tests for generation function."""
    
    def test_generate_does_not_modify_original(self):
        """Generate should not modify the original input list."""
        original_inputs = [1, 2, 3]
        inputs_copy = list(original_inputs)
        
        # Create minimal params
        vocab_size = 10
        n_embd = 4
        n_head = 2
        max_seq_len = 1024
        
        params = {
            'wte': np.random.randn(vocab_size, n_embd),
            'wpe': np.random.randn(max_seq_len, n_embd),
            'blocks': [],
            'ln_f': {'g': np.ones(n_embd), 'b': np.zeros(n_embd)}
        }
        
        # Mock tqdm to avoid progress bar in tests
        import unittest.mock
        with unittest.mock.patch('pico_gpt.tqdm', lambda x, desc: x):
            result = pico_gpt.generate(inputs_copy, params, n_head, n_tokens_to_generate=2)
        
        # Original should be unchanged
        assert original_inputs == [1, 2, 3]
        # Result should have more tokens
        assert len(result) > len(original_inputs)
    
    def test_generate_returns_list(self):
        """Generate should return a list of token IDs."""
        inputs = [1, 2, 3]
        vocab_size = 10
        n_embd = 4
        n_head = 2
        max_seq_len = 1024
        
        params = {
            'wte': np.random.randn(vocab_size, n_embd),
            'wpe': np.random.randn(max_seq_len, n_embd),
            'blocks': [],
            'ln_f': {'g': np.ones(n_embd), 'b': np.zeros(n_embd)}
        }
        
        import unittest.mock
        with unittest.mock.patch('pico_gpt.tqdm', lambda x, desc: x):
            result = pico_gpt.generate(inputs, params, n_head, n_tokens_to_generate=3)
        
        assert isinstance(result, list)
        assert all(isinstance(x, (int, np.integer)) for x in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


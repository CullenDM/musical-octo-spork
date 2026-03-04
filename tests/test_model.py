"""
Tests for model architecture components.

These tests run on CPU only (no CUDA required) and verify that:
  - ModelConfig validates correctly
  - LinearAccel produces the same result as nn.Linear (CPU fallback)
  - MultiHeadAttentionAccel produces outputs of the correct shape
  - Transformer forward pass produces logits / loss of the correct shape
  - Transformer backward pass runs without error
"""

import math
import pytest
import torch
import torch.nn as nn

from model.config import ModelConfig
from model.layers import LinearAccel, MultiHeadAttentionAccel
from model.transformer import Transformer


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.d_ff == 4 * cfg.d_model

    def test_custom_d_ff(self):
        cfg = ModelConfig(d_model=256, d_ff=512)
        assert cfg.d_ff == 512

    def test_invalid_heads(self):
        with pytest.raises(ValueError):
            ModelConfig(d_model=256, n_heads=7)

    def test_mixed_precision_flag(self):
        cfg = ModelConfig(use_mixed_precision=False)
        assert cfg.use_mixed_precision is False


# ---------------------------------------------------------------------------
# LinearAccel — CPU fallback must match nn.Linear
# ---------------------------------------------------------------------------

class TestLinearAccel:
    def setup_method(self):
        torch.manual_seed(0)

    def test_output_shape(self):
        layer = LinearAccel(32, 64)
        x = torch.randn(4, 8, 32)
        out = layer(x)
        assert out.shape == (4, 8, 64)

    def test_matches_nn_linear(self):
        """On CPU the fallback path must give the same result as nn.Linear."""
        lin = nn.Linear(16, 32, bias=True)
        accel = LinearAccel(16, 32, bias=True)
        # Copy weights so they are identical
        accel.weight.data.copy_(lin.weight.data)
        accel.bias.data.copy_(lin.bias.data)

        x = torch.randn(5, 16)
        torch.testing.assert_close(accel(x), lin(x))

    def test_no_bias(self):
        layer = LinearAccel(8, 16, bias=False)
        assert layer.bias is None
        x = torch.randn(3, 8)
        out = layer(x)
        assert out.shape == (3, 16)

    def test_backward(self):
        layer = LinearAccel(8, 16)
        x = torch.randn(4, 8, requires_grad=True)
        loss = layer(x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_reset_parameters(self):
        """Kaiming initialisation must produce weights with reasonable variance."""
        layer = LinearAccel(512, 512)
        std = layer.weight.data.std().item()
        assert 0.01 < std < 0.5


# ---------------------------------------------------------------------------
# MultiHeadAttentionAccel
# ---------------------------------------------------------------------------

class TestMultiHeadAttentionAccel:
    def setup_method(self):
        torch.manual_seed(42)

    def test_output_shape(self):
        attn = MultiHeadAttentionAccel(d_model=64, n_heads=4)
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert out.shape == (2, 10, 64)

    def test_with_padding_mask(self):
        attn = MultiHeadAttentionAccel(d_model=64, n_heads=4)
        x = torch.randn(2, 8, 64)
        # Mark last 2 tokens in first sequence as padding
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, -2:] = True
        out = attn(x, key_padding_mask=mask)
        assert out.shape == (2, 8, 64)

    def test_invalid_heads(self):
        with pytest.raises(ValueError):
            MultiHeadAttentionAccel(d_model=64, n_heads=5)

    def test_backward(self):
        attn = MultiHeadAttentionAccel(d_model=32, n_heads=4)
        x = torch.randn(2, 6, 32, requires_grad=True)
        loss = attn(x).sum()
        loss.backward()
        assert x.grad is not None

    def test_training_vs_eval_dropout(self):
        """Outputs should be deterministic in eval mode."""
        attn = MultiHeadAttentionAccel(d_model=32, n_heads=4, dropout=0.5)
        x = torch.randn(1, 4, 32)
        attn.eval()
        out1 = attn(x)
        out2 = attn(x)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class TestTransformer:
    @pytest.fixture
    def small_config(self):
        return ModelConfig(
            vocab_size=100,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            max_seq_len=16,
            dropout=0.0,
        )

    @pytest.fixture
    def model(self, small_config):
        torch.manual_seed(0)
        return Transformer(small_config)

    def test_logits_shape(self, model, small_config):
        tokens = torch.randint(0, small_config.vocab_size, (2, 8))
        logits = model(tokens)
        assert logits.shape == (2, 8, small_config.vocab_size)

    def test_loss_scalar(self, model, small_config):
        tokens = torch.randint(0, small_config.vocab_size, (2, 8))
        loss = model(tokens, targets=tokens)
        assert loss.shape == ()  # scalar
        assert loss.item() > 0

    def test_backward(self, model, small_config):
        tokens = torch.randint(0, small_config.vocab_size, (2, 8))
        loss = model(tokens, targets=tokens)
        loss.backward()
        # Every parameter with requires_grad should have a gradient
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_weight_tying(self, model):
        """LM head weight must be the same tensor as the token embedding."""
        assert model.lm_head.weight is model.token_emb.weight

    def test_causal_mask_shape(self, small_config):
        mask = Transformer._causal_mask(8, torch.device("cpu"))
        assert mask.shape == (8, 8)
        # Upper triangle should be -inf, lower (incl. diagonal) should be 0
        assert torch.isinf(mask[0, 1])
        assert mask[1, 0] == 0.0

    def test_different_batch_sizes(self, model, small_config):
        for bs in (1, 3, 8):
            tokens = torch.randint(0, small_config.vocab_size, (bs, 6))
            out = model(tokens)
            assert out.shape == (bs, 6, small_config.vocab_size)

    def test_padding_mask(self, model, small_config):
        tokens = torch.randint(0, small_config.vocab_size, (2, 8))
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, -2:] = True  # pretend last 2 tokens are padding
        logits = model(tokens, key_padding_mask=mask)
        assert logits.shape == (2, 8, small_config.vocab_size)

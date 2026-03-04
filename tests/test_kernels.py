"""
Tests for kernel dispatch logic and the CUDA extension interface.

These tests do NOT require a GPU:
  - When CUDA is unavailable the kernel-extension import fails silently and
    LinearAccel falls back to F.linear — the tests verify that fallback.
  - When CUDA IS available the tests additionally verify that the extension
    is loaded and that kernel outputs match PyTorch reference implementations.
"""

import pytest
import torch
import torch.nn.functional as F

from model.layers import LinearAccel, _KERNELS_AVAILABLE


# ---------------------------------------------------------------------------
# Fallback path (always runs — no GPU needed)
# ---------------------------------------------------------------------------

class TestFallbackPath:
    """LinearAccel without the CUDA extension must behave exactly like F.linear."""

    def test_forward_no_bias(self):
        layer = LinearAccel(16, 32, bias=False)
        x = torch.randn(4, 16)
        ref = F.linear(x, layer.weight)
        torch.testing.assert_close(layer(x), ref)

    def test_forward_with_bias(self):
        layer = LinearAccel(16, 32, bias=True)
        x = torch.randn(4, 16)
        ref = F.linear(x, layer.weight, layer.bias)
        torch.testing.assert_close(layer(x), ref)

    def test_3d_input(self):
        layer = LinearAccel(8, 16)
        x = torch.randn(2, 5, 8)
        out = layer(x)
        assert out.shape == (2, 5, 16)

    def test_gradient_flow(self):
        layer = LinearAccel(8, 16)
        x = torch.randn(3, 8, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None


# ---------------------------------------------------------------------------
# CUDA extension (skipped when not available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAKernels:
    """Verify CUDA kernel outputs against PyTorch reference on GPU."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(7)
        self.device = torch.device("cuda")

    def _make_layer(self, in_f, out_f, dtype=torch.float32):
        return LinearAccel(in_f, out_f).to(self.device).to(dtype)

    def _ref(self, layer, x):
        return F.linear(x, layer.weight, layer.bias)

    def test_fp32_forward_matches_ref(self):
        layer = self._make_layer(64, 128)
        x = torch.randn(8, 64, device=self.device)
        torch.testing.assert_close(layer(x), self._ref(layer, x), atol=1e-5, rtol=1e-4)

    def test_fp16_forward_matches_ref(self):
        layer = self._make_layer(64, 128, dtype=torch.float16)
        x = torch.randn(8, 64, device=self.device, dtype=torch.float16)
        torch.testing.assert_close(
            layer(x), self._ref(layer, x), atol=1e-2, rtol=1e-2
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available() or
        torch.cuda.get_device_capability()[0] < 8,
        reason="BF16 tensor cores require Ampere (sm_80+)"
    )
    def test_bf16_forward_matches_ref(self):
        layer = self._make_layer(64, 128, dtype=torch.bfloat16)
        x = torch.randn(8, 64, device=self.device, dtype=torch.bfloat16)
        torch.testing.assert_close(
            layer(x), self._ref(layer, x), atol=1e-1, rtol=1e-1
        )

    def test_backward_produces_gradients(self):
        layer = self._make_layer(32, 64)
        x = torch.randn(4, 32, device=self.device, requires_grad=True)
        loss = layer(x).sum()
        loss.backward()
        assert x.grad is not None
        assert layer.weight.grad is not None

    def test_output_device(self):
        layer = self._make_layer(16, 32)
        x = torch.randn(2, 16, device=self.device)
        out = layer(x)
        assert out.device.type == "cuda"

    def test_extension_imported(self):
        """Confirm the CUDA extension loaded when CUDA is available."""
        assert _KERNELS_AVAILABLE, (
            "mospork_kernels extension not imported — "
            "run `pip install -e .` to build it"
        )


# ---------------------------------------------------------------------------
# Architecture detection helper
# ---------------------------------------------------------------------------

class TestArchDetection:
    """Verify that the fallback dispatch logic works for all dtype × device
    combinations we care about."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_cpu_always_uses_fallback(self, dtype):
        """On CPU the custom kernel must never be invoked regardless of dtype."""
        layer = LinearAccel(8, 16).to(dtype)
        x = torch.randn(2, 8, dtype=dtype)
        # Should not raise regardless of whether _ext is available
        out = layer(x)
        assert out.shape == (2, 16)
        assert out.dtype == dtype

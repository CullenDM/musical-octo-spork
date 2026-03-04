"""
Accelerated layer implementations.

Each layer tries to dispatch to a custom CUDA kernel (``mospork_kernels``)
when:
  * the extension has been compiled and imported successfully, AND
  * the input tensors reside on a CUDA device.

When either condition is not met the layer falls back transparently to a
standard PyTorch implementation so the model can always run on CPU or on
systems where the extension has not been built yet.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# ---------------------------------------------------------------------------
# Optional CUDA extension
# ---------------------------------------------------------------------------
try:
    import mospork_kernels as _ext  # built by setup.py

    _KERNELS_AVAILABLE = True
except ImportError:
    _ext = None
    _KERNELS_AVAILABLE = False

# Additive value used to mask out attention positions (effectively -∞ in softmax).
_MASK_NEG_INF: float = -1e9


def _kernels_available_for(x: Tensor) -> bool:
    """Return True when kernels can be used for tensor *x*."""
    return _KERNELS_AVAILABLE and x.is_cuda


# ---------------------------------------------------------------------------
# Custom autograd functions that call the CUDA kernels
# ---------------------------------------------------------------------------


class _LinearAccelFunction(torch.autograd.Function):
    """Autograd wrapper around the custom linear forward/backward kernels."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:  # type: ignore[override]
        # x: (*, in_features), weight: (out_features, in_features)
        out = _ext.linear_forward(x, weight, bias)
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        x, weight = ctx.saved_tensors
        grad_x, grad_w, grad_b = _ext.linear_backward(
            grad_output, x, weight, ctx.has_bias
        )
        return grad_x, grad_w, grad_b


# ---------------------------------------------------------------------------
# nn.Module wrappers
# ---------------------------------------------------------------------------


class LinearAccel(nn.Module):
    """Drop-in replacement for :class:`torch.nn.Linear` with an optional CUDA
    kernel fast-path for FP16/BF16 inputs on Volta+ GPUs.

    Falls back to ``F.linear`` on CPU or when the CUDA extension is not
    available.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if _kernels_available_for(x):
            return _LinearAccelFunction.apply(x, self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class MultiHeadAttentionAccel(nn.Module):
    """Multi-head scaled dot-product attention with an optional CUDA kernel
    fast-path for the QKV projection and output projection.

    When the CUDA extension is unavailable, falls back to
    :func:`torch.nn.functional.scaled_dot_product_attention` (available in
    PyTorch ≥ 2.0) which itself uses flash-attention internally when
    applicable.

    Args:
        d_model: Total embedding dimension.
        n_heads: Number of attention heads.
        dropout: Attention dropout probability (applied during training).
        bias: Whether projection layers include a bias term.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Single fused QKV projection
        self.qkv_proj = LinearAccel(d_model, 3 * d_model, bias=bias)
        self.out_proj = LinearAccel(d_model, d_model, bias=bias)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: ``(batch, seq_len, d_model)``
            attn_mask: Optional additive mask ``(seq_len, seq_len)`` or
                ``(batch * n_heads, seq_len, seq_len)``.
            key_padding_mask: Optional boolean mask ``(batch, seq_len)``
                where ``True`` indicates positions to ignore.

        Returns:
            ``(batch, seq_len, d_model)``
        """
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        # (B, T, 3 * d_model) -> split into Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape to (B, H, T, D)
        def _reshape(t: Tensor) -> Tensor:
            return t.view(B, T, H, D).transpose(1, 2)

        q, k, v = _reshape(q), _reshape(k), _reshape(v)

        # Build combined mask for scaled_dot_product_attention
        mask = None
        if key_padding_mask is not None:
            # (B, 1, 1, T) — broadcast over heads and query positions
            mask = key_padding_mask[:, None, None, :].float() * _MASK_NEG_INF
        if attn_mask is not None:
            mask = attn_mask if mask is None else mask + attn_mask

        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p
        )

        # (B, H, T, D) -> (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(attn_out)

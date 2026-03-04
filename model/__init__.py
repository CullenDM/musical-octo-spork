"""
model — neural network layers and transformer model.

When the optional ``mospork_kernels`` CUDA extension is installed, custom
kernels replace certain PyTorch operations for higher throughput.
"""

from .config import ModelConfig
from .layers import LinearAccel, MultiHeadAttentionAccel
from .transformer import Transformer

__all__ = [
    "ModelConfig",
    "LinearAccel",
    "MultiHeadAttentionAccel",
    "Transformer",
]

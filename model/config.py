"""
Configuration dataclass for the Transformer model.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Hyper-parameters shared across the model.

    Args:
        vocab_size: Vocabulary size for the embedding layer.
        d_model: Hidden / embedding dimension.
        n_heads: Number of attention heads (must divide ``d_model`` evenly).
        n_layers: Number of transformer blocks.
        d_ff: Feed-forward inner dimension (defaults to ``4 * d_model``).
        max_seq_len: Maximum sequence length supported by positional encoding.
        dropout: Dropout probability used throughout the model.
        use_mixed_precision: Enable FP16/BF16 automatic mixed precision where
            supported.  Ignored when running on CPU.
        bias: Whether linear projections include a bias term.
    """

    vocab_size: int = 50_257
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: Optional[int] = None
    max_seq_len: int = 1_024
    dropout: float = 0.1
    use_mixed_precision: bool = True
    bias: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

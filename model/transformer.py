"""
Transformer model built from accelerated layers.

Architecture
------------
Token embedding + learned positional encoding
  └─ N × TransformerBlock
       ├─ LayerNorm
       ├─ MultiHeadAttentionAccel (causal mask)
       ├─ Residual
       ├─ LayerNorm
       ├─ Feed-forward (LinearAccel → GELU → LinearAccel)
       └─ Residual
  └─ Final LayerNorm
  └─ Language-model head (weight-tied to embedding)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .config import ModelConfig
from .layers import LinearAccel, MultiHeadAttentionAccel


class FeedForward(nn.Module):
    """Position-wise feed-forward network using :class:`LinearAccel`."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.fc1 = LinearAccel(d_model, d_ff, bias=bias)
        self.fc2 = LinearAccel(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(torch.nn.functional.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-norm."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttentionAccel(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.drop(self.attn(self.norm1(x), attn_mask, key_padding_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class Transformer(nn.Module):
    """Autoregressive transformer language model.

    Args:
        config: :class:`~model.config.ModelConfig` instance.

    Example::

        config = ModelConfig(vocab_size=50257, d_model=256, n_heads=4, n_layers=4)
        model = Transformer(config)
        tokens = torch.randint(0, config.vocab_size, (2, 32))
        logits = model(tokens)            # (2, 32, vocab_size)
        loss   = model(tokens, targets=tokens)  # scalar cross-entropy
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        # Weight-tied LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, LinearAccel)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Causal mask helper
    # ------------------------------------------------------------------

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
        """Return an additive causal mask of shape ``(seq_len, seq_len)``."""
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        targets: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            input_ids: Long tensor ``(batch, seq_len)``.
            targets: Optional long tensor ``(batch, seq_len)`` for computing
                cross-entropy loss.  When provided the method returns a scalar
                loss; otherwise it returns the logits.
            key_padding_mask: Boolean ``(batch, seq_len)`` where ``True``
                marks padding tokens.

        Returns:
            ``Tensor`` — either logits ``(batch, seq_len, vocab_size)`` or
            scalar cross-entropy loss when ``targets`` is supplied.
        """
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

        causal_mask = self._causal_mask(T, device)

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
            return loss

        return logits

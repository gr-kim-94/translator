from typing import Optional, Tuple

import torch
from torch import nn
from addnorm import AddNorm
from ffn import FFNAttention
from multihead_attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    """Self-attention + FFN 블록을 하나의 인코더 레이어로 묶은 모듈."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.attn_add_norm = AddNorm(d_model, dropout)

        self.feed_forward = FFNAttention(d_model, dropout)
        self.ffn_add_norm = AddNorm(d_model, dropout)


    def forward(
            self, 
            x: torch.Tensor, 
            padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        q = k = v = x
        attn_out, _ = self.self_attention(q, k, v, mask=padding_mask)
        residual = self.attn_add_norm(x, attn_out)

        ffn_out = self.feed_forward(residual)
        return self.ffn_add_norm(residual, ffn_out)

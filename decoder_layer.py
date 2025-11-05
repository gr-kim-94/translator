from typing import Optional

import torch
from torch import nn

from addnorm import AddNorm
from multihead_attention import MultiHeadAttention
from ffn import FFNAttention


class DecoderLayer(nn.Module):
    """Masked self-attention → cross-attention → FFN 흐름을 구현한 단일 디코더 레이어."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.self_add_norm = AddNorm(d_model, dropout)

        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_add_norm = AddNorm(d_model, dropout)

        self.feed_forward = FFNAttention(d_model, dropout)
        self.ffn_add_norm = AddNorm(d_model, dropout)


    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        '''
        x : tgt token embedding + positional encoding
        memory : encoder output
        self_mask : tgt 토크나이저 padding + decoder 이후 데이터 -\inf로 만든 mask
        memory_mask : encoder padding 포함된 mask와 tgt 토크나이저 padding 포함된 mask
        '''
        q = k = v = x
        # Decoder 1차 MultiHeadAttention
        self_attn_out, _ = self.self_attention(q, k, v, mask=self_mask)
        residual = self.self_add_norm(x, self_attn_out)

        # Encoder Output + Mask된 Decoder 1차 Output -> Decoder 2차 MultiHeadAttention
        cross_attn_out, _ = self.cross_attention(residual, memory, memory, mask=memory_mask)
        cross_residual = self.cross_add_norm(residual, cross_attn_out)
    
        ffn_out = self.feed_forward(cross_residual)
        return self.ffn_add_norm(cross_residual, ffn_out)
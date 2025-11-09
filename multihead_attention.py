import math
from typing import Optional, Tuple

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention as defined in Vaswani et al. (2017)."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
 
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k,v : (batch, num_heads, seq_len, head_dim)
        head_dim = query.size(-1)
        # 메모리 효율적인 행렬 곱셈
        # q_1 * k_2, q_1 * k_3,... 모든 토큰에대해 내곱을 해주는 부분.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        print("Scaled Scores shape:", scores.shape)

        # 결과: (batch, num_heads, seq_len, seq_len) -> (몇개의 문장인지, 몇개의 head인지, Q 토큰 위치, K 토큰 위치)
        if mask is not None:
            dim = mask.dim()
            print("Attention Mask Dim : ", dim, ", Attention Mask Shape : ", mask.shape)
            if dim == 3:
                mask = mask.unsqueeze(1)
            print("Attention New Mask Shape : ", mask.shape)
            scores = scores.masked_fill(mask == 0, float("-inf"))


        # softmax(Q * K^T / sqrt(d_k))
        # softmax 적용 (dim=-1 : 마지막 차원 기준으로 softmax 적용 -> 각 query의 attention weight 합이 1이 되게 만듦)
        # attention에선 dim을 -1로 주로 설정함. 이유는 마지막 차원이 key 토큰 위치에 해당하기 때문.
        # scores의 마지막 차원은 K 토큰 위치에 해당.
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # softmax(Q * K^T / sqrt(d_k)) * V
        output = torch.matmul(attn_weights, value)
        print("Scaled Dot Product Attention Output : ", output[0], "\n====\nAttention Weight : ", attn_weights[0])
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention module with learnable projections and output layer."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            # shape 사이즈가 같아야하기때문에
            # d_model을 num_heads로 나눠서 나머지가 0이 아니면 ValueError.
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # "Attention is All You Need" 논문에서는 Q, K, V 프로젝션에 bias를 사용 X.
        self.w_q = nn.Linear(d_model, d_model, bias=False) # bias : y = xW^T (+ b) (True면 + b)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.size()
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # batch -> seq_len -> num_heads -> head_dim 기존 순서를
        # batch -> num_heads -> seq_len -> head_dim 순서로 바꿔줘야함.
        # Attention 계산에서는 head 별로 병렬 연산을 해줘야하기 때문에 seq_len보다 num_heads가 먼저 있어야하기때문.
        return tensor.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

    def _combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        # 1️⃣ (batch, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = tensor.size()
        # contiguous 👉 메모리상에서 텐서를 연속된(continuous) 형태로 다시 정렬(copy)해주는 함수예요. transpose 다음에 view 해주려면 contiguous 해줘야함.
        tensor = tensor.transpose(1, 2).contiguous()
        # 2️⃣ 모든 head를 concat (flatten) -> (batch, seq_len, d_model)
        # d_model = num_heads * head_dim : 모든 head의 결과(head_dim)를 하나의 벡터(d_model)로 이어붙이는 것입니다
        return tensor.view(batch_size, seq_len, num_heads * head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        # scale dot product attention
        attn_output, attn_weights = self.attention(q, k, v, mask=mask)

        # concat attention
        attn_output = self._combine_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        
        # 서브레이어 출력에 dropout 적용
        # attn_output = self.dropout(attn_output)
        return attn_output, attn_weights
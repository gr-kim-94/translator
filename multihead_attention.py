#!/usr/bin/env python
# coding: utf-8

from typing import Tuple

import torch
from torch import nn
from transformers import AutoTokenizer

from attention_input import MultiHeadAttention
from transformer_modules import PositionalEncoding


# nn.Module 상속받았기때문에 forward 자동 호출
# 모듈 인스턴스를 함수처럼 호출할 때(output = module(input)) 내부적으로 nn.Module.__call__이 실행되고 그 안에서 forward가 호출됩니다.
class MultiHeadAttentionLayer(nn.Module):
    """End-to-end multi-head attention demo aligned with Vaswani et al. (2017)."""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        masked: bool = False,
        max_position_embeddings: int = 512,
    ) -> None:
        super().__init__()
        self.masked = masked        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model, max_position_embeddings
        )
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)

    def forward(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1️⃣ 토큰화
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        token_ids = tokens["input_ids"]
        print("토큰 목록:", self.tokenizer.tokenize(text))
        print("토큰 ID:", token_ids)
        for t in token_ids[0]:
            print(f"{t}\t -> {self.tokenizer.decode([t])}")    

        # 2️⃣ 임베딩
        embeddings = self.embedding(token_ids)
        # Positional Encoding + Embedding
        embeddings = self.positional_encoding(embeddings)

        # Decoder masking
        seq_len = embeddings.size(1)
        attn_mask = None
        if self.masked:
            attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=embeddings.device))

        # 3️⃣ Multi Head Attention
        q = k = v = embeddings
        attn_output, attn_weights = self.attention(q, k, v, mask=attn_mask)
        return attn_output, embeddings


if __name__ == "__main__":
    text = "I like coffee in the morning because it helps me wake up and stay focused."
    layer = MultiHeadAttentionLayer()
    output, embeddings = layer(text)
    print("Attention output shape:", output.shape)
    print("Attention weights shape:", embeddings.shape)

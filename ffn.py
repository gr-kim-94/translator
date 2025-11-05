from transformers import AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from addnorm import AddNorm
from multihead_attention import MultiHeadAttentionLayer

class FFNAttention(nn.Module):
    def __init__(self, 
                 d_model = 512,
                 drop_out_nd = 0.1):
        super().__init__()
        
        d_ff = 4*d_model # 내부 확장 차원 (보통 2048 = 4×d_model)
        self.linear = nn.Linear(d_model, d_ff)
        self.linear_t = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out_nd)

    def forward(self, residual):
        net = nn.Sequential(
            self.linear,
            self.relu,
            self.dropout,
            self.linear_t,
        )

        ffn_out = net(residual)
        print("FFN Output shape:", ffn_out.shape)

        return ffn_out

if __name__ == "__main__":
    drop_out_nd = 0.1

    text = "I like coffee in the morning because it helps me wake up and stay focused."
    mha = MultiHeadAttentionLayer(text) 
    print("==== mha.out : ", mha.out.shape) 
    addNorm = AddNorm(mha.X_input, mha.out, mha.d_model, drop_out_nd)

    ffn = FFNAttention(mha.out, addNorm.residual, mha.d_model)
    print(ffn.ffn_out)
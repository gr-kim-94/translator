from transformers import AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from addnorm import AddNorm
from multihead_attention import MultiHeadAttention

class FFNAttention:
    def __init__(self, out, residual, d_model):
        self.drop_out_nd = 0.1
        self.d_ff = 4*d_model # 내부 확장 차원 (보통 2048 = 4×d_model)

        net = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(self.drop_out_nd),
            nn.Linear(self.d_ff, d_model),
        )

        self.ffn_out = net(residual)
        print("FFN Output shape:", self.ffn_out.shape)

if __name__ == "__main__":
    drop_out_nd = 0.1

    text = "I like coffee in the morning because it helps me wake up and stay focused."
    mha = MultiHeadAttention(text) 
    print("==== mha.out : ", mha.out.shape) 
    addNorm = AddNorm(mha.X_input, mha.out, mha.d_model, drop_out_nd)

    ffn = FFNAttention(mha.out, addNorm.residual, mha.d_model)
    print(ffn.ffn_out)
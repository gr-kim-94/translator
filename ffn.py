import torch.nn as nn

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
            self.linear_t,
            # self.dropout
        )

        ffn_out = net(residual)
        print("FFN Output shape:", ffn_out.shape)

        return ffn_out

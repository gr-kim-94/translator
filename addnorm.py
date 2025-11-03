import torch.nn as nn

class AddNorm:
    def __init__(self, x, out, d_model, drop_out_nd):
        norm = nn.LayerNorm(d_model)
        dropout = nn.Dropout(drop_out_nd)

        # residual connection : 입력 + 서브레이어 출력
        # dropout : 과적합 방지를 위해 일부 뉴런을 랜덤하게 비활성화 : sublayer 출력에 적용
        # norm() : 합친 결과에 정규화 적용
        self.residual = norm(x + dropout(out))
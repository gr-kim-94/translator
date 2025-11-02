# drop out : 데이터의 특정규칙들을 제외하고 날리는 것.
drop_out_nd = 0.1 # 드롭아웃 비율, (overffitting 되었다면 0.1보다 더 높게 설정, underffitting 되었다면 0.1보다 더 낮게 설정)

def add_norm(x, out, d_model) -> torch.Tensor:
    norm = nn.LayerNorm(d_model)
    dropout = nn.Dropout(drop_out_nd)

    # residual connection : 입력 + 서브레이어 출력
    # dropout : 과적합 방지를 위해 일부 뉴런을 랜덤하게 비활성화 : sublayer 출력에 적용
    # norm() : 합친 결과에 정규화 적용
    residual = norm(x + dropout(out))

    return residual

def ffn(d_model, residual, net_output):
    d_ff = 4*d_model # 내부 확장 차원 (보통 2048 = 4×d_model)

    net = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Dropout(drop_out_nd),
        nn.Linear(d_ff, d_model),
    )

    net_output = net(residual)
    print("FFN Output shape:", net_output.shape)

    add_norm(residual, net_output, d_model)
    print("FFN Add & Norm shape:", type(residual), residual.shape) 

    return residual
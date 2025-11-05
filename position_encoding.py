import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with embedding scaling, as in Vaswani et al."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        '''
        max_len : token list의 최대 사이즈
        -> attention 연산은 토큰이 너무 많아지면 메모리 사용량도 n², 시간복잡도도 O(n²)라서 컴퓨터 부하가 심해짐.
        -> 너무 긴 문장은 어떻게 할 것인가?
           이건 청크(Chumk) 분할 처리해서 문장을 나눠 토큰화를 하거나 특수 모델을 사용한다.
        -> 이때 각 문장간의 길이를 맞추기 위해서도 사용하는게 padding 처리
        '''
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(                                # 지수 함수 (exponential)로 원래 비율 복원
            torch.arange(0, d_model, 2, dtype=torch.float32) # [0, 2, ...] -> 짝수에대한 값. d_model이 4라면 [0, 2]
            * (-math.log(10000.0) / d_model)                 # 로그 변환 
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32) # d_model 차원만큼 0으로 초기화된 행렬 생성
        pe[:, 0::2] = torch.sin(position * div_term)            # `0::2` : 짝수 차원 (0, 2, 4, …) -> sin 파형
        pe[:, 1::2] = torch.cos(position * div_term)            # `1::2` : 홀수 차원 (1, 3, 5, …) -> cos 파형
        self.register_buffer("pe", pe, persistent=False)        # register_buffer : PyTorch에서 중요한 메모리 관리 메커니즘. -> self.pe = pe로 대체할 수 있음.
                                                                # 모델의 파라미터는 아니지만(학습되지 않음), GPU에 올려두고 모델과 함께 사용해야 하는 텐서를 등록할 때 사용
                                                                # persistent : 모델을 저장할 때 이 버퍼는 저장 여부 결정
        self.d_model = d_model

        print("Positional Encoding : ", self.pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x : embedding된 토큰 리스트 Tensor
        '''
        # Positional Encoding + Embedding
        # x.shape : (batch_size, seq_len, d_model)
        # pe.shape : (seq_len, d_model)
        print("x shape : ", x.shape, ", pe shape : ", self.pe.shape)

        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1) 
        pe = self.pe[: seq_len].unsqueeze(0) # seq_len만큼 사용하기 위해. unsqueeze(0) : 0 - 추가할 차원의 위치. (batch_size, seq_len, d_model)
        print("x shape : ", x.shape, ", new pe shape : ", pe.shape)

        return x + pe

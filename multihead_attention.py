#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

class MultiHeadAttention:
    def __init__(self, text, masked = False):
        self.masked = masked

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.d_model = 4                         # 논문 기준은 512

        self.tokens = self.tokenizer_text(text)
        self.X = self.embedding_tokens(self.tokens)

        self.max_position = len(self.tokens["input_ids"][0])  # Maximum sequence length

        self.pe = self.positional_encoding()
        # [[0,1,2,3], [a,b,c,d], [e,f,g,h], ...] : 4개의 차원이 토큰 개수만큼 존재
        # 0번째 토큰의 PE 값들 : [0, a, e, ...]
        print("Positional Encoding : ", self.pe)

        # Positional Encoding + Embedding
        pe_tensor = torch.tensor(self.pe, dtype=self.X.dtype)
        print(self.X.shape, pe_tensor.shape)
        pe_tensor.unsqueeze_(0)  # 배치 차원 추가 -> (1, token_len, d_model)
        print(self.X.shape, pe_tensor.shape)

        self.X_input = self.X + pe_tensor
        print("X_input : ", self.X_input)

        self.output = self.attention_input(self.X_input)
        self.out = self.concat_attention(self.output)

    def tokenizer_text(self, text: str):
        # 1️⃣ 토큰화 + 숫자화
        text = "I like coffee in the morning because it helps me wake up and stay focused."
        tokens = self.tokenizer(text, return_tensors="pt") # pt : pytorch, tf : TensorFlow
        token_ids = tokens["input_ids"][0]

        print("토큰 목록:", self.tokenizer.tokenize(text))
        print("토큰 ID:", token_ids)
        for t in token_ids:
            print(f"{t}\t -> {self.tokenizer.decode([t])}")    

        return tokens

    def embedding_tokens(self, tokens):
        # 2️⃣ 임베딩
        vocab_size = self.tokenizer.vocab_size   # 약 30,000개 단어

        # Embedding : token을 tensor 타입으로 넣어야함. 
        embedding = nn.ModuleDict({
                "token_embedding" : nn.Embedding(vocab_size, self.d_model)
                })

        print(embedding.token_embedding)
        X = embedding.token_embedding(tokens["input_ids"])  # shape: (1, token_len, d_model)

        print("임베딩 벡터 :", X)  # torch
        print("임베딩 벡터 크기:", X.shape)  # torch.Size([1, 5, 4])

        return X


    def positional_encoding(self):
        position = np.arange(self.max_position)[:, np.newaxis] # [[0], [1], [2], ... , [max_position-1]]
        # The original formula pos / 10000^(2i/d_model) is equivalent to pos * (1 / 10000^(2i/d_model)).
        # I use the below version for numerical stability

        # `np.arange(0, d_model, 2)` : [0, 2, ...] -> 짝수에대한 값. d_model이 4라면 [0, 2]
        # `np.log(10000.0)`          : 로그 변환 
        # `np.exp(...)`              : 지수 함수 (exponential)로 원래 비율 복원
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))

        pe = np.zeros((self.max_position, self.d_model))     # d_model 차원만큼 0으로 초기화된 행렬 생성

        # `0::2` : 짝수 차원 (0, 2, 4, …) -> sin 파형
        # `1::2` : 홀수 차원 (1, 3, 5, …) -> cos 파형
        # `position * div_term` : 각 위치마다 주파수 스케일링 곱
        # 
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def attention_input(self, X_input):
        # 입력 토큰 임베딩을 Query 공간으로 선형 변환
        w_Q = nn.Linear(self.d_model, self.d_model) # y = xW^T + b
        w_K = nn.Linear(self.d_model, self.d_model)
        w_V = nn.Linear(self.d_model, self.d_model)

        # 선형변환으로 Q, K, V 생성
        Q = w_Q(X_input)  # shape: (1, token_len, d_model)
        K = w_K(X_input)
        V = w_V(X_input)

        # Q = Q.view(batch, seq_len, num_heads, head_dim)
        # batch : 한번의 학습에서 모델에 동시에 넣는 데이터 묶음. 1개면 토큰 1개, 32개면 토큰 32개.
        # num_heads : multi-head attention에서 head의 개수
        # default num_heads = 8 or 12,,, 여기선 8로 설정.

        print("Q shape:", Q.shape, "max_position : ", self.max_position)  # torch.Size([1, token_len, d_model]) : (배치 크기, 토큰 길이, 임베딩 차원) -> Q의 원소 수 : 1 * token_len * d_model
        num_heads = 2 # d_model은 반드시 num_heads로 나누어떨어져야 합니다.
        head_dim = self.d_model // num_heads  # 4 // 2 -> 2

        # view가 만들어내는 총 원소 수는 batch * seq_len * num_heads * head_dim -> Q의 원소 수와 동일해야한다.
        view_Q = Q.view(1, self.max_position, num_heads, head_dim)  # (batch, token_len, num_heads, head_dim)
        view_K = K.view(1, self.max_position, num_heads, head_dim)
        view_V = V.view(1, self.max_position, num_heads, head_dim)
        print("view_Q shape:", view_Q.shape)  # torch.Size([1, token_len, 1, d_model])

        # batch -> seq_len -> num_heads -> head_dim 순서를
        # batch -> num_heads -> seq_len -> head_dim 순서로 바꿔줘야함.
        # Attention 계산에서는 head 별로 병렬 연산을 해줘야하기 때문에 seq_len보다 num_heads가 먼저 있어야하기때문.
        transposed_Q = view_Q.transpose(1, 2)
        transposed_K = view_K.transpose(1, 2)
        transposed_V = view_V.transpose(1, 2)
        print("transposed_Q shape:", transposed_Q.shape) 

        scores = torch.matmul(transposed_Q, transposed_K.transpose(-2, -1))
        # 결과: (batch, num_heads, seq_len, seq_len) -> (몇개의 문장인지, 몇개의 head인지, Q 토큰 위치, K 토큰 위치)
        print("scores shape:", scores.shape)
        
        if self.masked:
            # decoder에서 적용해야되는 부분, scores.masked_fill(0) mask가 True인 위치를 value로 채워 넣어 무시되게 한다. 이전 데이터만 인식할 수 있도록!
            mask = torch.tril(torch.ones(self.max_position, self.max_position))  # 하삼각행렬 : 행렬의 대각선 아래쪽부분만 1로 남기고 나머지는 0으로 만들게 함.
            print(mask)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            print(scores)

        # Q * K^T / sqrt(d_k)
        scores = scores / math.sqrt(head_dim)
        print("Scaled Scores shape:", scores.shape)

        # softmax(Q * K^T / sqrt(d_k))
        # softmax 적용 (dim=-1 : 마지막 차원 기준으로 softmax 적용 -> 각 query의 attention weight 합이 1이 되게 만듦)
        # attention에선 dim을 -1로 주로 설정함. 이유는 마지막 차원이 key 토큰 위치에 해당하기 때문.
        # scores의 마지막 차원은 K 토큰 위치에 해당.
        attention_weights = torch.softmax(scores, dim=-1)
        print("Attention Weights shape :", attention_weights.shape)

        # softmax(Q * K^T / sqrt(d_k)) * V
        output = torch.matmul(attention_weights, transposed_V)
        print("Output shape:", output.shape)  # (batch, num_heads, seq_len, head_dim)

        return output

    def concat_attention(self, output):
        # 1️⃣ (batch, num_heads, seq_len, head_dim)
        # contiguous 👉 메모리상에서 텐서를 연속된(continuous) 형태로 다시 정렬(copy)해주는 함수예요. transpose 다음에 view 해주려면 contiguous 해줘야함.
        transposed_O = output.transpose(1, 2).contiguous()
        #  -> (batch, seq_len, num_heads, head_dim)

        batch, num_heads, seq_len, head_dim = output.shape

        # 2️⃣ 모든 head를 concat (flatten)
        out = transposed_O.view(batch, seq_len, self.d_model)
        #  -> (batch, seq_len, d_model)
        print("Final Output shape:", out.shape)  # (batch, seq_len, d_model)

        return out


if __name__ == "__main__":
    text = "I like coffee in the morning because it helps me wake up and stay focused."
    mha = MultiHeadAttention(text)  
    print(mha.out)
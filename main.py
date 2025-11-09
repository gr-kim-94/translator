from typing import Tuple

import torch
from torch import nn
from transformers import AutoTokenizer
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer
from position_encoding import PositionalEncoding

import pandas as pd

class TransformerModel(nn.Module):
    """ Start!
        Attention Is All You Need의 인코더-디코더 구조를 그대로 재현한 학습용 구현.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6, # 하이퍼파라미터!! dept를 얼마나 들어갈 것인가? 논문에서 6을 사용함.
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        vocab_size = self.tokenizer.vocab_size
        self.src_embedding = nn.Embedding(vocab_size, d_model) # 입력값에대한 embedding 객체
        self.tgt_embedding = nn.Embedding(vocab_size, d_model) # 출력값에대한 embedding 객체
        self.tgt_embedding_weight = self.tgt_embedding.weight
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_position_embeddings)

        self.encoder_layers = nn.ModuleList(
            # Encoder * N(num_layers)
            [EncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        self.decoder_layers = nn.ModuleList(
            # Decoder * N(num_layers)
            [DecoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d_model)

        self.generator = nn.Linear(d_model, vocab_size, bias=False)
        self.generator.weight = self.tgt_embedding_weight


    def _expand_padding_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len) → (batch, 1, 1, seq_len)."""
        return attention_mask.unsqueeze(1).unsqueeze(2)


    def _build_decoder_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        디코더 self attention 전용 마스크.
        causal mask와 padding mask를 결합해 (batch, seq_len, seq_len) 형태로 반환한다.
        """
        batch_size, seq_len = attention_mask.size()
        device = attention_mask.device

        # decoder에서 적용해야되는 부분, scores.masked_fill(0) mask가 True인 위치를 value로 채워 넣어 무시되게 한다. 이전 데이터만 인식할 수 있도록!
        causal = torch.tril(torch.ones((seq_len, seq_len), device=device))   # 하삼각행렬 : 행렬의 대각선 아래쪽부분만 1로 남기고 나머지는 0으로 만들게 함.
        valid_q = attention_mask.unsqueeze(1)  # (batch, 1, seq_len)
        valid_k = attention_mask.unsqueeze(2)  # (batch, seq_len, 1)

        combined = causal.unsqueeze(0) * valid_q * valid_k
        print("_build_decoder_mask => combined : ", combined.shape)
        return combined


    def _build_cross_attention_mask(
        self, tgt_attention_mask: torch.Tensor, src_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        tgt_attention_mask : 결과 text padding 값을 포함한 mask
        src_attention_mask : encoder padding 값을 포함한 mask

        (batch, tgt_len) X (batch, src_len) → (batch, tgt_len, src_len)
        디코더에서 패딩 토큰이 포함된 Query 또는 Key는 모두 차단한다.
        """
        tgt_valid = tgt_attention_mask.unsqueeze(2)
        src_valid = src_attention_mask.unsqueeze(1)
        return tgt_valid * src_valid

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,    # 배치 내 가장 긴 시퀀스까지 패드 처리 -> 문장의 길이가 모두 다르기때문에 통일 시켜주는 작업이 padding.
            #truncation=True,  # max_length를 초과하는 시퀀스를 잘라내기
            #max_length=10    # 최대 10개 토큰 길이로 잘라내거나 채우기
        )
        token_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        print("token_ids : ", token_ids, token_ids.shape)
        print("token_attention_mask : ", attention_mask, attention_mask.shape)

        x = self.src_embedding(token_ids)
        x = self.positional_encoding(x)
        
        padding_mask = self._expand_padding_mask(attention_mask)
        for layer in self.encoder_layers:
            # 이전 layer 출력값인 x를 다음 layer에서 사용함.
            x = layer(x, padding_mask)

        # 마지막 layer 층 출력만 사용함. 최총 문맥 표현.
        encoded = self.encoder_norm(x)
        return encoded, attention_mask

    def decode(
        self,
        text: str,
        memory: torch.Tensor,
        memory_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,    # 배치 내 가장 긴 시퀀스까지 패드 처리 -> 문장의 길이가 모두 다르기때문에 통일 시켜주는 작업이 padding.
            #truncation=True,  # max_length를 초과하는 시퀀스를 잘라내기
            #max_length=10    # 최대 10개 토큰 길이로 잘라내거나 채우기
        )
        token_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        x = self.tgt_embedding(token_ids)
        x = self.positional_encoding(x)

        self_mask = self._build_decoder_mask(attention_mask)
        cross_mask = self._build_cross_attention_mask(attention_mask, memory_attention_mask)

        for layer in self.decoder_layers:
            x = layer(x, memory, self_mask, cross_mask)

        decoded = self.decoder_norm(x)
        return decoded, attention_mask

    
    def forward(self, 
                src_text: str, 
                tgt_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # memory : encoder output K/V
        # memory_mask : encoder padding 여부를 알 수 있는 mask.
        memory, memory_mask = self.encode(src_text) 
        decoded, _ = self.decode(tgt_text, memory, memory_mask)

        logits = self.generator(decoded)
        return logits

    # def predict(
    #     self,
    #     src_texts,
    #     max_len=64,
    #     device=None,
    #     bos_token="[CLS]",
    #     eos_token="[SEP]",
    # ):
    #     device = device or next(self.parameters()).device
    #     self.eval()

    #     bos_id = self.tokenizer.convert_tokens_to_ids(bos_token)
    #     eos_id = self.tokenizer.convert_tokens_to_ids(eos_token)
        
    #     memory, memory_mask = self.encode(src_texts)
        
    #     batch_size = memory.size(0)
    #     generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    #     attn = torch.ones_like(generated)

    #     finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    #     for _ in range(max_len):
    #         logits = self.decode(
    #             memory,
    #             generated,
    #             src_attention_mask=memory_mask,
    #             tgt_attention_mask=attn,
    #         )
    #         next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    #         generated = torch.cat([generated, next_token], dim=1)
    #         attn = torch.ones_like(generated)

    #         finished |= next_token.squeeze(1) == eos_id
    #         if finished.all():
    #             break

    #     texts = tokenizer.batch_decode(generated.tolist(), skip_special_tokens=True)
    #     return texts


# if __name__ == "__main__":
#     path = "./data/en_fr_small.tsv"
#     df = pd.read_csv(path, sep="\t")

#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = TransformerModel(d_model=128, num_heads=8, num_layers=2)#.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
#     pad_id = model.tokenizer.pad_token_id
#     loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

#     # 같은 데이터셋을 3번 학습시킨다.
#     for epoch in range(3):
#         model.train()
#         for i in range(len(df)):
#             batch = df.iloc[i]
#             print(batch)
#             src = model.tokenizer(batch["en"], return_tensors="pt", padding=True)#.to(device)
#             tgt = model.tokenizer(batch["fr"], return_tensors="pt", padding=True)#.to(device)

#             decoder_inputs = tgt["input_ids"][:, :-1]
#             decoder_mask = tgt["attention_mask"][:, :-1]
#             labels = tgt["input_ids"][:, 1:]

#             logits = model(
#                 src["input_ids"],
#                 decoder_inputs,
#                 src_attention_mask=src["attention_mask"],
#                 tgt_attention_mask=decoder_mask,
#             )
#             loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
    
if __name__ == "__main__":
    src = "I like coffee in the morning because it helps me wake up and stay focused."
    tgt = "J'aime le café le matin car il m'aide à me réveiller."
    tf = TransformerModel(d_model=128, num_heads=8, num_layers=2)
    logits = tf(src, tgt)
    print("Result output :", logits, logits.shape)

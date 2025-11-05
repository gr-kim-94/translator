# main.py
from multihead_attention import MultiHeadAttentionLayer
from addnorm import AddNorm
from ffn import FFNAttention
from torch import nn

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.d_model = 4
        self.num_heads = 2
        self.drop_out_nd = 0.1


    def input_encoder(self, text):
        mha = MultiHeadAttentionLayer(
            d_model= self.d_model, 
            num_heads= self.num_heads, 
            dropout= self.drop_out_nd
            )
        attn_output, embeddings = mha(text)  # forward 호출
        print("==== mha.out : ", attn_output.shape) # attn_output, attn_weights, token_ids
        addNorm = AddNorm(self.d_model, self.drop_out_nd)
        residual = addNorm(
            embeddings,
            attn_output
        )
        
        ffn = FFNAttention(self.d_model, self.drop_out_nd)
        ffn_out = ffn(residual)
        print(ffn_out)
        ffn_addNorm = AddNorm(self.d_model, self.drop_out_nd)
        ffn_residual = ffn_addNorm(residual, ffn_out)
        print("FFN Add & Norm shape:", type(ffn_residual), ffn_residual.shape)
        
        return ffn_residual

    # def decode(self, target_text, encoder_out):
        # decoder = DecoderLayer(self.d_model, self.drop_out_nd)
        # return decoder(target_text, encoder_out)

    def forward(self, text):
        input_out = self.input_encoder(text)
        return input_out

if __name__ == "__main__":
    text = "I like coffee in the morning because it helps me wake up and stay focused."
    tf = TransformerModel()
    input_out = tf(text)
    print(input_out)
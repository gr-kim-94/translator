# main.py
from multihead_attention import MultiHeadAttention
from addnorm import AddNorm
from ffn import FFNAttention

class TransformerModel:
    def __init__(self, text):
        self.d_model = 4
        self.drop_out_nd = 0.1

        self.input_out = self.input_encoder(text)

    def input_encoder(self, text):
        mha = MultiHeadAttention(text) 
        print("==== mha.out : ", mha.out.shape) 
        addNorm = AddNorm(mha.X_input, mha.out, mha.d_model, self.drop_out_nd)

        ffn = FFNAttention(mha.out, addNorm.residual, mha.d_model)
        print(ffn.ffn_out)
        ffn_addNorm = AddNorm(addNorm.residual, ffn.ffn_out, self.d_model, self.drop_out_nd)
        print("FFN Add & Norm shape:", type(ffn_addNorm.residual), ffn_addNorm.residual.shape)

        return ffn_addNorm.residual


if __name__ == "__main__":
    text = "I like coffee in the morning because it helps me wake up and stay focused."
    tf = TransformerModel(text)
    print(tf.input_out)
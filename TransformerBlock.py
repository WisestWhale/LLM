import torch
import torch.nn  as nn  
import LayerNorm
from SelfAttention import MultiHeadAttention , CasualAttention
import FeedForward
class TransformerBlock(nn.Module):
    def __init__(self , d_model  , max_length  , num_heads):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model ,d_model , max_length, num_heads  )
        self.norm2 = LayerNorm(d_model)
        self.ff() = FeedForward(d_model)
        
    def Forward(self , x):
        x = x + self.attn(self.norm1(x))
        x + x + self.ff(self.norm2(x))
        return x
        
    
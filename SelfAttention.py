import torch 
import torch.nn as nn


class CasualAttention(nn.Module):
    def __init__(self , d_in , d_out, max_length):
        super().__init__()
        self.W_k = nn.Linear(d_in , d_out , bias=False)
        self.W_v = nn.Linear(d_in , d_out , bias=False)
        self.W_q = nn.Linear(d_in , d_out , bias=False)  
    
        self.register_buffer(
            "mask",
             torch.triu(torch.ones(max_length, max_length), diagonal=1).bool()
        )
    def forward(self, x):
        batch_size, seq_len, d_in = x.shape 

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        scores = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ v 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, max_length, num_heads):
        super().__init__()
        head_dim = d_out // num_heads
        self.heads = nn.ModuleList([
            CasualAttention(d_in, head_dim, max_length)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        return torch.cat(outputs, dim=-1)    

x = torch.randn(1, 3, 12)
mha  = MultiHeadAttention(12 , 15 , 10 , 3)
print(mha(x))
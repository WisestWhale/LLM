import torch 
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self , d_in , d_out, max_length):
        super().__init__()
        self.W_k = nn.Linear(d_in , d_out , bias=False)
        self.W_v = nn.Linear(d_in , d_out , bias=False)
        self.W_q = nn.Linear(d_in , d_out , bias=False)  
    
        self.register_buffer(
            "mask",
             torch.triu(torch.ones(max_length, max_length), diagonal=1).bool()
        )
    def Forward(self , x):
        
        batch_size, seq_len, d_in = x.shape
        
        q = self.W_k(x)
        v = self.W_v(x)
        k = self.W_v(x)
        scores = q @ k.transpose(-2 , -1) / (k.shape[-1] ** 0.5)
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len], float("-inf"))
        weights = torch.softmax(scores  , dim = -1)
        return weights @ v


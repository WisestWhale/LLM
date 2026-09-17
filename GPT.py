import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock
from LayerNorm import LayerNorm


class GPT(nn.Module):
    def __init__(self , vocab_size , d_model , max_length , num_heads , num_blocks):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size , d_model)
        self.pos_emb = nn.Embedding(max_length , d_model)
        
        self.blocks = nn.ModuleList([TransformerBlock(d_model , max_length , num_heads)
        for _ in range(num_blocks)])
        
        
        
        self.final_norm = LayerNorm(d_model)
        self.out_head = nn.Linear(d_model , vocab_size , bias = False)
        
        
    def forward(self, ids):
        batch_size, seq_len = ids.shape

        positions = torch.arange(seq_len, device=ids.device)
        x = self.token_emb(ids) + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.out_head(x)
    
    
ids = torch.randint(0, 50, (1, 3))
model = GPT(vocab_size=50, d_model=12, max_length=10, num_heads=3, num_blocks=2)
print(model(ids).shape)
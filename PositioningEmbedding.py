import torch 
import torch.nn as nn


class TokenEmbeddingPosiotioning(nn.modules):
    def __init__(self , vocab_size , output_dim , max_length):
        super.__init__()
        self.token_embedding = nn.Embedding(vocab_size , output_dim)
        self.pos_embedding = nn.Embedding(max_length ,output_dim)
        
    def forward(self, batch_inputs):
        batch_size, seq_len = batch_inputs.shape
        positions = torch.arange(seq_len, device=batch_inputs.device)
        token_embeds = self.token_embedding(batch_inputs)
        pos_embeds = self.pos_embedding(positions)
        return token_embeds + pos_embeds
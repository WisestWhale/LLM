import torch
import torch.nn as nn
class FeedForward(nn.Module):
    def __init__(self , d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model , 4 * d_model)
        self.gelu = nn.GELU()
        self.fc2  = nn.Linear(d_model* 4 , d_model)
    
    
    def forward(self , x):
         x = self.fc1(x)
         x = self.gelu(x)
         x = self.fc2(x)
         return x
     
x  = torch.randn(1 , 3 ,12)
ff = FeedForward(12)
print(ff(x))
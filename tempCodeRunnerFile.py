import torch, torch.nn as nn
t2 = nn.Embedding(10, 4)
p = torch.arange(3)
print(p)
print(t2(p))
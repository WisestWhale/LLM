import torch
import torch.nn.functional as F
from GPT import GPT
from DataLoader import dataloader ,tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(
    vocab_size=len(tokenizer.word_to_id),
    d_model=128,
    max_length=64,
    num_heads=4,
    num_blocks=4,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

for epoch in range(50):
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            targets.flatten()
        )

        loss.backward()
        optimizer.step()

    print(f"epoch {epoch}  loss {loss.item():.4f}")
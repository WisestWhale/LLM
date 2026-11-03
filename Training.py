import torch
import torch.nn.functional as F
from GPT import GPT
from DataLoader import dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"
# Same architecture Generate.py loads, so the saved checkpoint is usable there.
model = GPT(
    vocab_size=50257,
    d_model=128,
    max_length=128,
    num_heads=4,
    num_blocks=4,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

for epoch in range(50):
    total_loss = 0.0
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
        total_loss += loss.item()

    print(f"epoch {epoch}  mean loss {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "model.pt")

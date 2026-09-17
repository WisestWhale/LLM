import torch
import torch.nn.functional as F
from GPT import GPT
from DataLoader import dataloader, tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(
    vocab_size=50257,
    d_model=128,
    max_length=128,
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
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch}  loss {loss.item():.4f}")

torch.save(model.state_dict(), "model.pt")




@torch.no_grad()
def generate(model, ids, max_new_tokens, max_length, temperature=1.0):
    model.eval()

    for _ in range(max_new_tokens):
        ids_cropped = ids[:, -max_length:]

        logits = model(ids_cropped)
        logits = logits[:, -1, :]

        probs = torch.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        ids = torch.cat([ids, next_id], dim=1)

    return ids


start = tokenizer.encode("the")
ids = torch.tensor([start]).to(device)

out = generate(model, ids, max_new_tokens=50, max_length=64)
print(tokenizer.decode(out[0].tolist()))
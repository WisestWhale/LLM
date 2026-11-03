import torch
import torch.nn.functional as F
import tiktoken
from torch.utils.data import DataLoader
from GPT import GPT
from SlidingWindow import GPTDatasetV1

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = tiktoken.get_encoding("gpt2")

# all 4 books, split 90% train / 10% validation
books = ["the-verdict.txt", "pride.txt", "frankenstein.txt", "sherlock.txt"]
text = "\n\n".join(open(f, encoding="utf-8").read() for f in books)
split = int(len(text) * 0.9)
train_set = GPTDatasetV1(text[:split], tokenizer, max_length=128, stride=64)
val_set   = GPTDatasetV1(text[split:], tokenizer, max_length=128, stride=128)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=16)

model = GPT(vocab_size=tokenizer.n_vocab, d_model=128, max_length=128,
            num_heads=4, num_blocks=4).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

def get_loss(x, y):
    logits = model(x.to(device))
    return F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten())

best_val = float("inf")

for epoch in range(5):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = get_loss(x, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = sum(get_loss(x, y).item() for x, y in val_loader) / len(val_loader)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "model_small.pt")
        print(f"epoch {epoch}  train {loss.item():.3f}  val {val_loss:.3f}  <- saved")
    else:
        print(f"epoch {epoch}  train {loss.item():.3f}  val {val_loss:.3f}")
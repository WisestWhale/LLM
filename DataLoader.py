import torch
import urllib.request
import tiktoken
from torch.utils.data import DataLoader
from SlidingWindow import GPTDatasetV1

books = {
    "the-verdict.txt": "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
    "pride.txt": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "frankenstein.txt": "https://www.gutenberg.org/files/84/84-0.txt",
    "sherlock.txt": "https://www.gutenberg.org/files/1661/1661-0.txt",
}

parts = []
for filename, url in books.items():
    try:
        urllib.request.urlretrieve(url, filename)
    except Exception as e:
        print(f"skipped {filename}: {e}")
        continue
    with open(filename, "r", encoding="utf-8") as f:
        parts.append(f.read())

text = "\n\n".join(parts)
print(f"characters: {len(text):,}")

tokenizer = tiktoken.get_encoding("gpt2")
print(f"tokens: {len(tokenizer.encode(text)):,}")

dataset = GPTDatasetV1(text, tokenizer, max_length=128, stride=64)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
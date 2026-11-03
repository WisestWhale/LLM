"""Measure how well a saved checkpoint predicts a text file (loss and perplexity)."""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from GPT import GPT


@torch.no_grad()
def evaluate(model, token_ids, max_length, max_windows=None):
    """Average next-token loss over non-overlapping windows of the text."""
    if len(token_ids) < 2:
        raise ValueError("Text must contain at least two tokens.")
    was_training = model.training
    model.eval()
    try:
        total_loss = 0.0
        total_tokens = 0
        starts = range(0, len(token_ids) - 1, max_length)
        for count, start in enumerate(starts):
            if max_windows is not None and count >= max_windows:
                break
            chunk = token_ids[start : start + max_length + 1]
            inputs = torch.tensor([chunk[:-1]])
            targets = torch.tensor([chunk[1:]])
            logits = model(inputs.to(next(model.parameters()).device))
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten().to(logits.device),
                                   reduction="sum")
            total_loss += loss.item()
            total_tokens += targets.numel()
        mean_loss = total_loss / total_tokens
        return mean_loss, math.exp(mean_loss), total_tokens
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=Path, help="UTF-8 text file to score.")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).with_name("model.pt"))
    parser.add_argument("--max-windows", type=int,
                        help="Score only the first N windows of 128 tokens (faster on CPU).")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.max_windows is not None and args.max_windows < 1:
        parser.error("--max-windows must be positive.")
    try:
        text = args.text.read_text(encoding="utf-8-sig")
    except OSError as error:
        parser.error(f"Could not read text: {error}")

    import tiktoken

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(vocab_size=50257, d_model=128, max_length=128, num_heads=4, num_blocks=4).to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    except (RuntimeError, ValueError) as error:
        parser.error(f"Checkpoint does not match this model: {error}")
    token_ids = tiktoken.get_encoding("gpt2").encode(text)
    try:
        loss, perplexity, tokens = evaluate(model, token_ids, 128, args.max_windows)
    except ValueError as error:
        parser.error(str(error))
    print(f"tokens scored: {tokens:,}")
    print(f"loss: {loss:.4f}")
    print(f"perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()

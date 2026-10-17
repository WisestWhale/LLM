"""Generate text from a saved checkpoint without running training."""

import argparse
from pathlib import Path

import torch
from GPT import GPT


@torch.no_grad()
def generate(model, ids, max_new_tokens, max_length, temperature=1.0, top_k=None, top_p=1.0):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if max_new_tokens < 0 or max_length < 1:
        raise ValueError("Token count must be nonnegative and context length positive.")
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError("Provide a nonempty batch of token IDs.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive.")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be between zero (exclusive) and one.")

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits = model(ids[:, -max_length:])[:, -1, :] / temperature
            if top_k is not None:
                cutoff = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            if top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                # Keep the token that crosses the threshold, and always keep one.
                remove = cumulative_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                mask = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
                logits = logits.masked_fill(mask, float("-inf"))
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).with_name("model.pt"))
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Cumulative sampling probability (0 < p <= 1).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("Prompt must not be empty.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.tokens < 0 or args.temperature <= 0 or args.top_k < 1:
        parser.error("Use nonnegative tokens, positive temperature and positive top-k.")

    if not 0 < args.top_p <= 1:
        parser.error("Use top-p between zero (exclusive) and one.")

    import tiktoken

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Match the architecture used by the original Generate.py training script.
    model = GPT(vocab_size=50257, d_model=128, max_length=128, num_heads=4, num_blocks=4).to(device)
    try:
        weights = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(weights)
    except (RuntimeError, ValueError) as error:
        parser.error(f"Checkpoint does not match this model: {error}")
    tokenizer = tiktoken.get_encoding("gpt2")
    ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    output = generate(model, ids, args.tokens, 128, args.temperature, args.top_k, args.top_p)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()

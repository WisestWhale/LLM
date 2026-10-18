# Small GPT language model

A learning project implementing a decoder-only transformer in PyTorch: token and position embeddings, causal multi-head attention, feed-forward layers and residual connections.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Generate text

```powershell
python Generate.py --prompt "The morning" --tokens 50 --temperature 0.8 --top-k 20 --top-p 0.9 --seed 42
```

Generation loads `model.pt`; it does not train or download books. The checkpoint must match the original generation architecture: GPT-2 vocabulary (50,257 tokens), width 128, context 128, four heads and four blocks. The GPT-2 tokenizer downloads its vocabulary on first use, so the first run needs internet access. Only load trusted checkpoints.

`--temperature` controls sampling randomness; `--top-k` limits each draw to the highest-scoring tokens. `--top-p` (nucleus sampling) keeps the smallest set of likely tokens whose combined probability reaches the chosen threshold, including the token that crosses it. It runs after top-k filtering; the default `1.0` disables this extra filter. A fixed seed helps repeat a run on the same device and software stack. Output quality depends on training; this is a small experimental model, not a chat assistant.

To save the generated text (including your prompt) to a file:

```powershell
python Generate.py --prompt "The morning" --tokens 50 --output story.txt
```

The text also appears in the terminal. Choose a new filename each time; existing files are not overwritten. The destination folder must already exist.

## Tests

```powershell
python -B -m unittest test_model -v
```

Seven CPU-friendly tests check output shape, feed-forward gradients, causal masking, context limits, generation settings, nucleus filtering and prompt preservation. They use a tiny random model and do not download data or train the saved checkpoint.

## Files

- `GPT.py`: model embeddings, transformer blocks and output projection.
- `TransformerBlock.py`, `SelfAttention.py`, `FeedForward.py`, `LayerNorm.py`: model components.
- `Generate.py`: checkpoint-based generation and command-line options.
- `DataLoader.py`, `SlidingWindow.py`: text loading and next-token training windows.
- `Training.py`: earlier training experiment; it currently expects a word-tokenizer interface while DataLoader uses tiktoken, so it needs alignment before use.
- `Tokenizer.py`: separate word-level tokenizer experiment.

## Recent fixes

The feed-forward residual is now assigned back into the block output, allowing feed-forward weights to receive gradients. Invalid attention dimensions and context lengths produce clear errors. Component examples run only when their files are executed directly.

The existing checkpoint predates the residual fix. It loads with the same parameter shapes, but output behavior changes because the feed-forward path is now active. Retraining is needed to assess the corrected architecture's quality; no retraining or benchmark claim is included here.

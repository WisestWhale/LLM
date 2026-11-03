# Small GPT language model

A learning project implementing a decoder-only transformer in PyTorch: token and position embeddings, causal multi-head attention, feed-forward layers and residual connections.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

```powershell
python train_small.py
```

Trains for 50 epochs on the four bundled books (90% train, 10% validation) and prints train and validation loss after each epoch. `model_small.pt` is overwritten only when validation loss improves, so the saved checkpoint is the best epoch, not the last. Every run starts from random weights, so copy the current `model_small.pt` first if you want to keep it. A CUDA GPU is used automatically when PyTorch can see one; on a CPU, 50 epochs is slow. The included `model_small.pt` was trained for 5 epochs.

## Generate text

```powershell
python Generate.py --prompt "The morning" --tokens 50 --temperature 0.8 --top-k 20 --top-p 0.9 --seed 42
```

Generation loads `model_small.pt` (pass `--checkpoint` to use another file); it does not train or download books. The checkpoint must match this architecture: GPT-2 vocabulary (50,257 tokens), width 128, feed-forward width 7 × 128, context 128, four heads and four blocks. The GPT-2 tokenizer downloads its vocabulary on first use, so the first run needs internet access. Only load trusted checkpoints.

`--temperature` controls sampling randomness; `--top-k` limits each draw to the highest-scoring tokens. `--top-p` (nucleus sampling) keeps the smallest set of likely tokens whose combined probability reaches the chosen threshold, including the token that crosses it. It runs after top-k filtering; the default `1.0` disables this extra filter. A fixed seed helps repeat a run on the same device and software stack. Output quality depends on training; this is a small experimental model, not a chat assistant.

To save the generated text (including your prompt) to a file:

```powershell
python Generate.py --prompt "The morning" --tokens 50 --output story.txt
```

The text also appears in the terminal. Choose a new filename each time; existing files are not overwritten. The destination folder must already exist.

To continue text saved in a UTF-8 file:

```powershell
python Generate.py --prompt-file story.txt --tokens 50 --output continued.txt
```

Use either `--prompt` or `--prompt-file`. Files can contain multiple lines. Add `--completion-only` to print and save only the newly generated text:

```powershell
python Generate.py --prompt-file story.txt --completion-only --output new_text.txt
```

Without that flag, output still includes the starting text. Keep the output filename different from the input filename.

## Evaluate a checkpoint

Score how well `model_small.pt` predicts a text file:

```powershell
python Evaluate.py held_out.txt --max-windows 50
```

It prints the average next-token loss and perplexity (lower is better) over non-overlapping 128-token windows. The bundled books are training data, so use a different text for a fair number, and the same file and `--max-windows` when comparing two checkpoints. Without `--max-windows` the whole file is scored.

## Tests

```powershell
python -B -m unittest discover -p "test_*.py" -v
```

CPU-friendly model tests check output shape, feed-forward gradients, causal masking, context limits, generation settings, nucleus filtering and prompt preservation. They use a tiny random model and do not download data or train the saved checkpoint. CLI tests use mocked model output to check prompt files, errors and completion-only output.

## Files

- `GPT.py`: model embeddings, transformer blocks and output projection.
- `TransformerBlock.py`, `SelfAttention.py`, `FeedForward.py`, `LayerNorm.py`: model components.
- `Generate.py`: checkpoint-based generation and command-line options.
- `Evaluate.py`: loss and perplexity of a checkpoint on a text file.
- `DataLoader.py`, `SlidingWindow.py`: text loading and next-token training windows.
- `train_small.py`: trains on the four bundled books with a 90/10 train/validation split and saves `model_small.pt` whenever validation loss improves.
- `Training.py`: simpler loop without validation; downloads missing books and saves `model.pt` after 50 epochs.
- `Tokenizer.py`: separate word-level tokenizer experiment.

## Recent fixes

The feed-forward residual is now assigned back into the block output, allowing feed-forward weights to receive gradients. Invalid attention dimensions and context lengths produce clear errors. Component examples run only when their files are executed directly.

The feed-forward layer is now 7 × the model width (was 4 ×), and `model_small.pt` was trained with it after the residual fix. The earlier `model.pt` used the old shapes and was removed. On a short excerpt of `sherlock.txt`, `Evaluate.py` gives a perplexity of about 434 for `model_small.pt` against about 1067 for the old checkpoint; that excerpt is likely part of the training split, so use unseen text for a fair comparison.

# Repository Guidelines

Brief guide for contributing to this tinyLLM repo so updates stay consistent and easy to review.

## Project Structure & Module Organization
- `model.py`: minimal decoder-only transformer with RMSNorm, Rotary Embedding, GQA, and MoE; runnable `main` prints loss and verifies backward on dummy data.
- `train.py`: sampling utility that expects a checkpoint at `out/ckpt.pt` plus matching `configurator.py` and `GPT/GPTConfig` definitions; adjust paths before use.
- `data/`: `gen_data.py` downloads Tiny Shakespeare, trains a ByteLevelBPETokenizer, and writes `bbpe/`, `train.bin`, and `val.bin`; `input.txt` caches the raw text.
- `t.ipynb`: scratch experiments; `README.md`: quickstart notes.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` (recommended) then `pip install torch tokenizers tiktoken requests numpy`.
- `python data/gen_data.py`: refresh dataset and tokenizer files.
- `python model.py`: smoke-test forward/backward (prefers MPS when available).
- `python train.py`: generate samples from a saved checkpoint in `out/`; set `device` as needed and ensure model definitions align with the checkpoint.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; classes use `CamelCase`, functions/variables use `snake_case`.
- Keep modules small and readable; add docstrings or brief comments when behavior is non-obvious.
- Prefer type hints for new public functions and avoid hard-coded paths—pass via arguments or config.

## Testing Guidelines
- No formal suite yet; at minimum run `python model.py` after changes.
- When adding features, add `pytest` cases under `tests/` (name files `test_*.py`), seed randomness, and cover forward shapes, losses, and tokenizer outputs.

## Commit & Pull Request Guidelines
- Existing commits are short Chinese summaries (e.g., “一次小改进”); keep messages concise (<72 chars) and action-oriented.
- For PRs: include a short intent summary, commands run, and notes on data or regenerated artifacts; link related issues when available and attach sample logs/output for model changes.

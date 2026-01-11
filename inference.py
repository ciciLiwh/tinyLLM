import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from model import DecoderOnlyTransformer, select_device


DATA_DIR = Path("data")
VOCAB_PATH = DATA_DIR / "bbpe" / "vocab.json"
MERGES_PATH = DATA_DIR / "bbpe" / "merges.txt"
CKPT_PATH = Path("out") / "decoder_epoch1.pt"
MAX_NEW_TOKENS = 100
TEMPERATURE = 1.0
TOP_K = 50  # set to None to disable


def load_vocab_size(vocab_path: Path) -> int:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return len(vocab)


def load_model(ckpt_path: Path, device: torch.device) -> Tuple[DecoderOnlyTransformer, int, int]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    vocab_size = checkpoint["vocab_size"]
    cfg = checkpoint["config"]
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=cfg["dim"],
        num_layers=cfg["num_layers"],
        num_q_heads=cfg["num_q_heads"],
        num_kv_heads=cfg["num_kv_heads"],
        moe_hidden=cfg["moe_hidden"],
        num_experts=cfg["num_experts"],
        max_seq_len=cfg["max_seq_len"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab_size, cfg["max_seq_len"]


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def sample(
    model: DecoderOnlyTransformer,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    max_seq_len: int,
) -> str:
    ids: List[int] = tokenizer.encode(prompt).ids
    for _ in range(max_new_tokens):
        # ensure context window fits model
        input_ids = ids[-max_seq_len:]
        x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]
        logits = model(x)[:, -1, :]  # last token logits
        logits = logits / max(temperature, 1e-6)
        logits = top_k_filter(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)
    return tokenizer.decode(ids)


def main() -> None:
    device = select_device()
    print("Using device:", device)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}. Train with train.py first.")

    vocab_size = load_vocab_size(VOCAB_PATH)
    tokenizer = ByteLevelBPETokenizer(str(VOCAB_PATH), str(MERGES_PATH))

    model, _, max_seq_len = load_model(CKPT_PATH, device)

    prompt = "To be, or not to be"
    print(f"Prompt: {prompt}")
    output = sample(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        max_seq_len=max_seq_len,
    )
    print("Generated text:\n", output)


if __name__ == "__main__":
    main()

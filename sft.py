import json
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tokenizers import ByteLevelBPETokenizer

from model import DecoderOnlyTransformer, select_device


# =========================
# Paths & hyperparameters
# =========================

DATA_PATH = Path("data") / "sft.jsonl"
VOCAB_PATH = Path("data") / "bbpe" / "vocab.json"
MERGES_PATH = Path("data") / "bbpe" / "merges.txt"
BASE_CKPT = Path("out") / "decoder_epoch1.pt"
OUT_DIR = Path("out")

BATCH_SIZE = 16
NUM_EPOCHS = 64
LEARNING_RATE = 5e-4
LOG_INTERVAL = 20
SEED = 42
IGNORE_INDEX = -100


# =========================
# Dataset
# =========================

class SFTDataset(Dataset):
    """Loads instruction/response pairs from jsonl and tokenizes with BPE."""

    def __init__(
        self,
        path: Path,
        tokenizer: ByteLevelBPETokenizer,
        eos_id: int,
    ) -> None:
        super().__init__()

        if not path.exists():
            raise FileNotFoundError(
                f"SFT file not found: {path}. "
                "Expect jsonl with fields instruction/input/output."
            )

        self.samples: List[List[int]] = []
        self.tokenizer = tokenizer
        self.eos_id = eos_id

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)

                prompt = build_prompt(
                    instruction=record.get("instruction", ""),
                    inp=record.get("input", ""),
                    output=record.get("output", ""),
                )

                ids = tokenizer.encode(prompt).ids
                ids.append(self.eos_id)
                self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> List[int]:
        return self.samples[idx]


def build_prompt(instruction: str, inp: str, output: str) -> str:
    if inp:
        return (
            f"Instruction:\n{instruction}\n\n"
            f"Input:\n{inp}\n\n"
            f"Output:\n{output}"
        )
    return f"Instruction:\n{instruction}\n\nOutput:\n{output}"


# =========================
# Collate
# =========================

def collate_fn(
    batch: List[List[int]],
    max_seq_len: int,
    pad_id: int,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    - 输入：x = [B, T]
    - 目标：y = [B, T]
    - causal LM: predict next token
    """

    input_seqs: List[torch.Tensor] = []
    target_seqs: List[torch.Tensor] = []

    for ids in batch:
        # 保证不会超过 RoPE 的 max_seq_len
        trimmed = ids[: max_seq_len]

        # x: [0 ... T-2]
        # y: [1 ... T-1]
        x = torch.tensor(trimmed[:-1], dtype=torch.long)
        y = torch.tensor(trimmed[1:], dtype=torch.long)

        input_seqs.append(x)
        target_seqs.append(y)

    inputs = pad_sequence(
        input_seqs,
        batch_first=True,
        padding_value=pad_id,
    )

    targets = pad_sequence(
        target_seqs,
        batch_first=True,
        padding_value=ignore_index,
    )

    return inputs, targets


# =========================
# Model loading
# =========================

def load_model(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[DecoderOnlyTransformer, int]:
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found at {ckpt_path}. "
            "Train with train.py first."
        )

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
    return model, vocab_size


# =========================
# Evaluation
# =========================

def evaluate(
    model: DecoderOnlyTransformer,
    data_loader: DataLoader,
    device: torch.device,
    vocab_size: int,
    ignore_index: int,
) -> float:
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=ignore_index,
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += (y != ignore_index).sum().item()

    model.train()
    return total_loss / max(total_tokens, 1)


# =========================
# Main
# =========================

def main() -> None:
    torch.manual_seed(SEED)

    device = select_device()
    print("Using device:", device)

    tokenizer = ByteLevelBPETokenizer(
        str(VOCAB_PATH),
        str(MERGES_PATH),
    )
    eos_id = tokenizer.token_to_id("<|endoftext|>") or 0

    # ---- load model ----
    model, vocab_size = load_model(BASE_CKPT, device)
    model.train()

    # ✅ 核心修复：严格使用模型的 RoPE max_seq_len
    max_seq_len = model.rope.cos.size(0)
    print(f"Using max_seq_len = {max_seq_len}")

    dataset = SFTDataset(DATA_PATH, tokenizer, eos_id)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(
            b,
            max_seq_len,
            eos_id,
            IGNORE_INDEX,
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(NUM_EPOCHS):
        print(f"SFT epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=IGNORE_INDEX,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f"step {batch_idx + 1}: loss {loss.item():.4f}")

            global_step += 1

        val_loss = evaluate(
            model,
            train_loader,
            device,
            vocab_size,
            IGNORE_INDEX,
        )

        ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
        print(f"epoch {epoch + 1} avg loss {val_loss:.4f}, ppl {ppl:.2f}")

    # 只在训练结束后保存最后一次checkpoint
    ckpt_path = OUT_DIR / "sft_final.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": vocab_size,
            "config": {
                "dim": model.embed.embedding_dim,
                "num_layers": len(model.layers),
                "num_q_heads": model.layers[0].attn.num_q_heads,
                "num_kv_heads": model.layers[0].attn.num_kv_heads,
                "moe_hidden": model.layers[0].moe.experts[0][0].out_features // 2,
                "num_experts": len(model.layers[0].moe.experts),
                "max_seq_len": max_seq_len,
            },
        },
        ckpt_path,
    )

    print(f"Saved final SFT checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

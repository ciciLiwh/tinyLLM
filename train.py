import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import DecoderOnlyTransformer, select_device


DATA_DIR = Path("data")
VOCAB_PATH = DATA_DIR / "bbpe" / "vocab.json"
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"

BLOCK_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 2
LEARNING_RATE = 3e-4
LOG_INTERVAL = 100
OUT_DIR = Path("out")
SEED = 1337


class BinaryDataset(Dataset):
    """Loads uint16 token streams and returns input/target pairs of length block_size."""

    def __init__(self, bin_path: Path, block_size: int) -> None:
        super().__init__()
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = np.asarray(
            self.data[idx: idx + self.block_size + 1],
            dtype=np.int64,
        )
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def load_vocab_size(vocab_path: Path) -> int:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return len(vocab)


def build_model(vocab_size: int) -> DecoderOnlyTransformer:
    # Match the lightweight demo config from model.py
    return DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=128,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        moe_hidden=256,
        num_experts=2,
        max_seq_len=max(BLOCK_SIZE, 64),
    )


def evaluate(
    model: DecoderOnlyTransformer,
    data_loader: DataLoader,
    device: torch.device,
    vocab_size: int,
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
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    return total_loss / total_tokens


def main() -> None:
    torch.manual_seed(SEED)
    device = select_device()
    print("Using device:", device)

    vocab_size = load_vocab_size(VOCAB_PATH)
    print("Vocab size:", vocab_size)

    train_ds = BinaryDataset(TRAIN_BIN, BLOCK_SIZE)
    val_ds = BinaryDataset(VAL_BIN, BLOCK_SIZE)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )

    model = build_model(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (global_step + 1) % LOG_INTERVAL == 0:
                print(f"step {global_step + 1}: loss {loss.item():.4f}")

            global_step += 1

        val_loss = evaluate(model, val_loader, device, vocab_size)
        print(f"validation loss: {val_loss:.4f}")

        ckpt_path = OUT_DIR / f"decoder_epoch{epoch + 1}.pt"
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
                    "max_seq_len": model.rope.cos.size(0),
                },
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

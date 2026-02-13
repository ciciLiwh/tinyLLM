import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from model import DecoderOnlyTransformer, select_device


# =========================
# Paths
# =========================

VOCAB_PATH = Path("data") / "bbpe" / "vocab.json"
MERGES_PATH = Path("data") / "bbpe" / "merges.txt"

# CKPT_PATH = Path("out") / "decoder_latest.pt"
# CKPT_PATH = Path("out") / "sft_latest.pt"
CKPT_PATH = Path("out") / "grpo_latest.pt"

# =========================
# Prompt
# =========================

def build_prompt(
    instruction: str,
    inp: Optional[str] = None,
) -> str:
    if inp:
        return (
            f"Instruction:\n{instruction}\n\n"
            f"Input:\n{inp}\n\n"
            f"Output:\n"
        )
    return f"Instruction:\n{instruction}\n\nOutput:\n"


# =========================
# Model loading
# =========================

def load_model(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[DecoderOnlyTransformer, int]:
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

    return model, vocab_size


# =========================
# Streaming generation
# =========================

@torch.no_grad()
def generate_stream(
    model: DecoderOnlyTransformer,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
):
    """
    逐 token 生成，边生成边 yield 字符串增量
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.token_to_id("<|endoftext|>") or 0

    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(
        input_ids, dtype=torch.long, device=device
    )[None, :]

    max_seq_len = model.rope.cos.size(0)

    # 已输出的 token 数（用于增量 decode）
    prev_len = input_ids.size(1)

    for _ in range(max_new_tokens):
        if input_ids.size(1) >= max_seq_len:
            break

        logits = model(input_ids)
        next_logits = logits[:, -1, :]

        if temperature <= 0:
            next_id = torch.argmax(next_logits, dim=-1)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

        input_ids = torch.cat(
            [input_ids, next_id[:, None]],
            dim=1,
        )

        # ===== 流式 decode =====
        decoded = tokenizer.decode(
            input_ids[0, prev_len:].tolist()
        )
        prev_len = input_ids.size(1)

        yield decoded

        if next_id.item() == eos_id:
            break


# =========================
# Main
# =========================

def main() -> None:
    device = select_device()
    print("Using device:", device)

    tokenizer = ByteLevelBPETokenizer(
        str(VOCAB_PATH),
        str(MERGES_PATH),
    )

    model, _ = load_model(CKPT_PATH, device)

    print("Model loaded.")
    print("RoPE max_seq_len:", model.rope.cos.size(0))
    print("-" * 60)

    while True:
        instruction = input("Instruction (empty to quit): ").strip()
        if not instruction:
            break

        inp = input("Input (optional): ").strip()
        inp = inp if inp else None

        prompt = build_prompt(instruction, inp)

        print("\n--- Model output (streaming) ---")

        for chunk in generate_stream(
            model,
            tokenizer,
            prompt,
            max_new_tokens=128,
            temperature=0.8,
        ):
            print(chunk, end="", flush=True)

        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
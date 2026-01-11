import json
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import ByteLevelBPETokenizer

from model import DecoderOnlyTransformer, select_device


# Paths and data
PROMPTS_PATH = Path("data") / "grpo_prompts.jsonl"
VOCAB_PATH = Path("data") / "bbpe" / "vocab.json"
MERGES_PATH = Path("data") / "bbpe" / "merges.txt"
SFT_CKPT = Path("out") / "sft_epoch1.pt"
OUT_DIR = Path("out")

# Hyperparameters
MAX_SEQ_LEN = 256
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.8
TOP_K = 50  # set to None to disable
GROUP_SIZE = 4  # generations per prompt
BATCH_PROMPTS = 2  # prompts per batch
NUM_EPOCHS = 2
LEARNING_RATE = 1e-5
BETA_KL = 0.1
ENTROPY_COEF = 0.01
GRAD_CLIP = 1.0
LOG_INTERVAL = 10
SEED = 1234


class PromptDataset(Dataset):
    """Reads prompts (instruction + optional input) from jsonl."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {path}. Expect jsonl with fields instruction/input."
            )
        self.prompts: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                prompt = build_prompt(
                    instruction=record.get("instruction", ""),
                    inp=record.get("input", ""),
                )
                self.prompts.append(prompt)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]


def build_prompt(instruction: str, inp: str) -> str:
    if inp:
        return f"Instruction:\n{instruction}\n\nInput:\n{inp}\n\nOutput:\n"
    return f"Instruction:\n{instruction}\n\nOutput:\n"


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def load_model(ckpt_path: Path, device: torch.device) -> Tuple[DecoderOnlyTransformer, int]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SFT checkpoint not found at {ckpt_path}. Run sft.py first.")
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


def generate_one(
    model: DecoderOnlyTransformer,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    max_seq_len: int,
) -> Tuple[List[int], List[int], str]:
    """Generate a single completion and return prompt ids, generated ids, and text."""
    prompt_ids = tokenizer.encode(prompt).ids
    prompt_ids = prompt_ids[-(max_seq_len - 1):]  # keep room for at least one token
    generated: List[int] = []
    ids = prompt_ids.copy()
    for _ in range(max_new_tokens):
        input_ids = ids[-max_seq_len:]
        x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]
        logits = model(x)[:, -1, :]  # last step logits
        logits = logits / max(temperature, 1e-6)
        logits = top_k_filter(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)
        generated.append(next_id)
    text = tokenizer.decode(ids)
    return prompt_ids, generated, text


def compute_reward(text: str) -> float:
    """
    Simple heuristic reward to illustrate GRPO wiring.
    Customize this function for task-specific signals (e.g., classifier scores).
    """
    length_reward = min(len(text.strip()) / 200.0, 1.0)
    return float(length_reward)


def compute_logprob_on_generated(
    model: DecoderOnlyTransformer,
    ref_model: DecoderOnlyTransformer,
    tokens: List[int],
    prompt_len: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute policy logprobs, ref logprobs (no grad), and entropy over generated positions.
    Returns: logprob_sum, kl_mean, entropy_mean
    """
    if len(tokens) < 2:
        raise ValueError("Need at least 2 tokens to compute logprobs.")
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device)[None, ...]
    targets = torch.tensor(tokens[1:], dtype=torch.long, device=device)[None, ...]

    logits = model(input_ids)
    logprobs = F.log_softmax(logits, dim=-1)

    with torch.no_grad():
        ref_logits = ref_model(input_ids)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)

    gen_start = max(prompt_len - 1, 0)
    t = torch.arange(targets.size(1), device=device)
    logp_taken = logprobs[0, t, targets[0]]
    ref_taken = ref_logprobs[0, t, targets[0]]

    logp_gen = logp_taken[gen_start:]
    ref_gen = ref_taken[gen_start:]

    kl = (logp_gen - ref_gen).mean()

    # entropy over generated positions
    probs_gen = logp_gen.exp()
    entropy = -(probs_gen * logp_gen).mean()

    return logp_gen.sum(), kl, entropy


def main() -> None:
    torch.manual_seed(SEED)
    device = select_device()
    print("Using device:", device)

    tokenizer = ByteLevelBPETokenizer(str(VOCAB_PATH), str(MERGES_PATH))
    model, vocab_size = load_model(SFT_CKPT, device)
    ref_model, _ = load_model(SFT_CKPT, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    dataset = PromptDataset(PROMPTS_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_PROMPTS, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"GRPO epoch {epoch + 1}/{NUM_EPOCHS}")
        for batch_idx, prompts in enumerate(loader):
            batch_loss = 0.0
            batch_count = 0
            for prompt in prompts:
                generations: List[Tuple[List[int], List[int], str]] = []
                rewards: List[float] = []
                for _ in range(GROUP_SIZE):
                    with torch.no_grad():
                        prompt_ids, gen_ids, text = generate_one(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            device=device,
                            max_new_tokens=MAX_NEW_TOKENS,
                            temperature=TEMPERATURE,
                            top_k=TOP_K,
                            max_seq_len=MAX_SEQ_LEN,
                        )
                    generations.append((prompt_ids, gen_ids, text))
                    rewards.append(compute_reward(text))

                rewards_tensor = torch.tensor(rewards, device=device)
                adv = rewards_tensor - rewards_tensor.mean()
                if rewards_tensor.std() > 1e-6:
                    adv = adv / rewards_tensor.std()

                loss = 0.0
                for i, (prompt_ids, gen_ids, _) in enumerate(generations):
                    tokens = (prompt_ids + gen_ids)[:MAX_SEQ_LEN]
                    logp_sum, kl, entropy = compute_logprob_on_generated(
                        model=model,
                        ref_model=ref_model,
                        tokens=tokens,
                        prompt_len=len(prompt_ids),
                        vocab_size=vocab_size,
                        device=device,
                    )
                    loss += -(adv[i] * logp_sum) + BETA_KL * kl - ENTROPY_COEF * entropy

                loss = loss / GROUP_SIZE
                loss.backward()
                batch_loss += loss.item()
                batch_count += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = batch_loss / max(batch_count, 1)
                print(f"step {batch_idx + 1}: loss {avg_loss:.4f}")

            global_step += 1

        ckpt_path = OUT_DIR / f"grpo_epoch{epoch + 1}.pt"
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
        print(f"Saved GRPO checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

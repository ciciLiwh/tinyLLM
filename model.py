import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    """Root mean square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        scaled = x * torch.rsqrt(norm + self.eps)
        return self.weight * scaled


# class RotaryEmbedding(nn.Module):
#     """Precomputes rotary positional embeddings."""

#     def __init__(self, dim: int, max_seq_len: int = 2048) -> None:
#         super().__init__()
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
#         t = torch.arange(max_seq_len).float()
#         freqs = torch.einsum("i,j->ij", t, inv_freq)
#         self.register_buffer("cos", freqs.cos())
#         self.register_buffer("sin", freqs.sin())

#     def apply_rope(self, x: Tensor, seq_len: int) -> Tensor:
#         cos = self.cos[:seq_len][None, :, None, :]
#         sin = self.sin[:seq_len][None, :, None, :]

#         x1, x2 = x[..., ::2], x[..., 1::2]
#         return torch.cat(
#             [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
#             dim=-1,
#         )
    
class RotaryEmbedding(nn.Module):
    """Precomputes rotary positional embeddings."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048) -> None:  # 改为head_dim
        super().__init__()
        # head_dim应该是每个头的维度
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def apply_rope(self, x: Tensor, seq_len: int) -> Tensor:
        cos = self.cos[:seq_len][None, :, None, :]  # [1, seq_len, 1, head_dim/2]
        sin = self.sin[:seq_len][None, :, None, :]  # [1, seq_len, 1, head_dim/2]
        
        # x的形状: [batch, seq_len, num_heads, head_dim]
        # 将head_dim分成两部分
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]  # 前一半
        x2 = x[..., half_dim:]  # 后一半
        
        return torch.cat(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
            dim=-1,
        )



class GQASelfAttention(nn.Module):
    """Grouped-query attention with per-head gating."""

    def __init__(self, dim: int, num_q_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_q_heads
        self.qNorm = RMSNorm(self.head_dim)
        self.kNorm = RMSNorm(self.head_dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.head_dim * num_kv_heads)
        self.v_proj = nn.Linear(dim, self.head_dim * num_kv_heads)
        self.gate_proj = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rope: RotaryEmbedding) -> Tensor:
        batch, seq_len, channels = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q = rope.apply_rope(q, seq_len)
        k = rope.apply_rope(k, seq_len)

        q = self.qNorm(q)
        k = self.kNorm(k)

        # GQA: repeat kv heads
        repeat = self.num_q_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

        attn = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.head_dim)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        attn = attn.masked_fill(causal_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", attn, v)
        gate = torch.sigmoid(self.gate_proj(out))
        out = out * gate
        out = out.reshape(batch, seq_len, channels)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU activation: (xW1) * sigmoid(xW2)."""

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


class MoE(nn.Module):
    """Simple top-1 routed mixture of experts."""

    def __init__(self, dim: int, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim * 2),
                SwiGLU(),
                nn.Linear(hidden_dim, dim),
            ) for _ in range(num_experts)
        ])

    def forward(self, x: Tensor) -> Tensor:
        _, seq_len, _ = x.shape
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)

        top1 = probs.argmax(dim=-1)
        out = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            mask = top1 == i
            if mask.any():
                out[mask] = expert(x[mask])

        return out


class DecoderBlock(nn.Module):
    """Decoder block with GQA and MoE."""

    def __init__(
        self,
        dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        moe_hidden: int,
        num_experts: int,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = GQASelfAttention(dim, num_q_heads, num_kv_heads)

        self.ln2 = RMSNorm(dim)
        self.moe = MoE(dim, moe_hidden, num_experts)

    def forward(self, x: Tensor, rope: RotaryEmbedding) -> Tensor:
        x = x + self.attn(self.ln1(x), rope)
        x = x + self.moe(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 6,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
        moe_hidden: int = 2048,
        num_experts: int = 4,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab_size, dim)

        # 计算每个头的维度
        head_dim = dim // num_q_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len)  # 传入head_dim

        self.layers = nn.ModuleList([
            DecoderBlock(dim, num_q_heads, num_kv_heads, moe_hidden, num_experts)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, self.rope)
        x = self.norm(x)
        return self.lm_head(x)


def select_device() -> torch.device:
    """Prefer MPS when present; fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_dummy_pass(
    vocab_size: int = 100,
    batch_size: int = 2,
    seq_len: int = 134,
) -> Tuple[Tensor, float]:
    device = select_device()
    print("Using device:", device)

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=256,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        moe_hidden=256,
        num_experts=2,
        max_seq_len=256,
    ).to(device)

    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len),
        device=device,
    )

    logits = model(input_ids)
    print("logits shape:", logits.shape)  # (B, T, vocab)

    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size),
        input_ids[:, 1:].reshape(-1),
    )
    print("loss:", loss.item())

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Backward & step OK ✅")
    return logits, loss.item()


def main() -> None:
    run_dummy_pass()


if __name__ == "__main__":
    main()

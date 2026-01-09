import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: (B, T, C)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x
    
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def apply_rope(self, x, seq_len):
        cos = self.cos[:seq_len][None, :, None, :]
        sin = self.sin[:seq_len][None, :, None, :]

        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin,
                          x1 * sin + x2 * cos], dim=-1)

class GQASelfAttention(nn.Module):
    def __init__(self, dim, num_q_heads, num_kv_heads):
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

    def forward(self, x, rope: RotaryEmbedding):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        q = rope.apply_rope(q, T)
        k = rope.apply_rope(k, T)

        q = self.qNorm(q)
        k = self.kNorm(k)

        # GQA: repeat kv heads
        repeat = self.num_q_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

        attn = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.head_dim)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        attn = attn.masked_fill(causal_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", attn, v)
        gate = torch.sigmoid(self.gate_proj(out))
        out = out * gate
        out = out.reshape(B, T, C)
        return self.out_proj(out)
    
class MoE(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts):
        super().__init__()
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, T, C = x.shape
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
    def __init__(self, dim, num_q_heads, num_kv_heads, moe_hidden, num_experts):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = GQASelfAttention(dim, num_q_heads, num_kv_heads)

        self.ln2 = RMSNorm(dim)
        self.moe = MoE(dim, moe_hidden, num_experts)

    def forward(self, x, rope):
        x = x + self.attn(self.ln1(x), rope)
        x = x + self.moe(self.ln2(x))
        return x
    
class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=512,
        num_layers=6,
        num_q_heads=8,
        num_kv_heads=2,
        moe_hidden=2048,
        num_experts=4,
        max_seq_len=2048
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, dim)
        self.rope = RotaryEmbedding(dim // num_q_heads, max_seq_len)

        self.layers = nn.ModuleList([
            DecoderBlock(dim, num_q_heads, num_kv_heads, moe_hidden, num_experts)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, self.rope)
        x = self.norm(x)
        return self.lm_head(x)
    
def main():
    # ====== device (MPS 优先) ======
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # ====== 超小模型，防止 M 芯片炸显存 ======
    vocab_size = 100
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        dim=128,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        moe_hidden=256,
        num_experts=2,
        max_seq_len=64
    ).to(device)

    # ====== dummy input ======
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len),
        device=device
    )

    # ====== forward ======
    logits = model(input_ids)
    print("logits shape:", logits.shape)  # (B, T, vocab)

    # ====== language modeling loss ======
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size),
        input_ids[:, 1:].reshape(-1)
    )
    print("loss:", loss.item())

    # ====== backward ======
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Backward & step OK ✅")


if __name__ == "__main__":
    main()
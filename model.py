"""A small GPT in plain PyTorch. Yours to modify or replace entirely —
attention, SSM, whatever — as long as evaluate.py still works and the
parameter cap holds.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256      # byte-level tokenizer default
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = True    # REQUIRED at BPE vocab 4096: unties -> ~2.57M > cap; tied -> ~1.9M
    qk_norm = False       # RMS-normalize q,k per head before attention (LR stabilizer)
    relu2 = False         # squared-ReLU MLP instead of GELU
    rmsnorm = False       # RMSNorm instead of LayerNorm for the block/final norms
    rope = False          # rotary position embeddings; replaces learned pos_emb (param-negative)


class RMSNorm(nn.Module):
    """RMS normalization over the last dim, with a learnable per-channel gain."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.qk_norm = getattr(cfg, "qk_norm", False)
        if self.qk_norm:
            self.qn = RMSNorm(self.head_dim)
            self.kn = RMSNorm(self.head_dim)
        self.rope = getattr(cfg, "rope", False)
        if self.rope:
            assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"
            inv_freq = 1.0 / (10000 ** (
                torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, q, k, T, device):
        t = torch.arange(T, device=device).float()
        freqs = torch.outer(t, self.inv_freq)            # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)          # (T, head_dim)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q, k = self.qn(q), self.kn(k)
        if self.rope:
            q, k = self._apply_rope(q, k, T, x.device)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.relu2 = getattr(cfg, "relu2", False)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x).square() if self.relu2 else F.gelu(x)
        return self.drop(self.proj(x))


def make_norm(cfg):
    return RMSNorm(cfg.n_embd) if getattr(cfg, "rmsnorm", False) \
        else nn.LayerNorm(cfg.n_embd)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = SelfAttention(cfg)
        self.ln2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.rope = getattr(cfg, "rope", False)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if not self.rope:                          # RoPE replaces the learned table
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = make_norm(cfg)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        # baseline init: plain normal, one std for everything
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if not self.rope:                          # RoPE injects position inside attention
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

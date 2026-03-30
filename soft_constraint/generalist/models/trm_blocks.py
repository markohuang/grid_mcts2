"""Building blocks for TRM iterative refinement: transformer and UNet variants.

FiLM conditioning:
    use_film=True  → gamma * x + beta on residual stream (non-skippable)
    use_film=False → additive projection inside residual branch (original, skippable)
"""
import math
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn import flash_attn_qkvpacked_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False


# =============================================================================
# Shared primitives
# =============================================================================

def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    return get_2d_sincos_pos_embed_rect(embed_dim, grid_size, grid_size)


def get_2d_sincos_pos_embed_rect(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    assert embed_dim % 4 == 0
    grid = torch.stack(torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32), indexing='ij'
    ), dim=-1).reshape(-1, 2)
    omega = 1.0 / (10000 ** (torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)))
    pos_h, pos_w = grid[:, 0:1] * omega, grid[:, 1:2] * omega
    return torch.cat([torch.sin(pos_h), torch.cos(pos_h), torch.sin(pos_w), torch.cos(pos_w)], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float = 4.0, dropout: float = 0.0):
        super().__init__()
        intermediate = int(hidden_size * expansion)
        self.w1 = nn.Linear(hidden_size, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x): return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, hidden_size // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        if HAS_FLASH and x.is_cuda:
            out = flash_attn_qkvpacked_func(qkv, dropout_p=self.dropout if self.training else 0.0)
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            ).transpose(1, 2)
        return self.proj(out.reshape(B, L, D))


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        L_ctx = context.shape[1]
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, L_ctx, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: gamma * x + beta (non-skippable).
    Handles both sequence (B, L, D) and spatial (B, C, H, W) tensors.
    """
    def __init__(self, cond_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_size))
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)
        self.proj[1].bias.data[:hidden_size] = 1.0  # gamma=1, beta=0 at init

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        if x.dim() == 4:
            gamma, beta = gamma[:, :, None, None], beta[:, :, None, None]
        else:
            gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
        return gamma * x + beta


class GridEmbedding(nn.Module):
    """Token + 2D sincos positional embedding for grid inputs."""
    def __init__(self, vocab_size: int, hidden_size: int, grid_size: int):
        super().__init__()
        self.vocab_size, self.seq_len = vocab_size, grid_size * grid_size
        self.scale = math.sqrt(hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size + 1, hidden_size)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        pos = get_2d_sincos_pos_embed(hidden_size, grid_size)
        self.register_buffer('pos_embed', torch.cat([pos, pos], dim=0))

    def forward(self, tokens: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        tok_emb = self.embed_tokens(tokens.flatten(1).clamp(0, self.vocab_size))
        return self.scale * (tok_emb + self.pos_embed[pos_offset:pos_offset + self.seq_len])

    def forward_soft(self, probs: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """Embed soft probability vectors via weighted sum of token embeddings."""
        tok_emb = probs @ self.embed_tokens.weight                  # (B, 81, V+1) @ (V+1, d) -> (B, 81, d)
        return self.scale * (tok_emb + self.pos_embed[pos_offset:pos_offset + self.seq_len])


# =============================================================================
# Transformer
# =============================================================================

class TransformerBlock(nn.Module):
    """Norm-then-residual block. Optional cross-attention and FiLM.
    When cond_dim > 0: FiLM modulates the residual stream (non-skippable).
    When cond_dim = 0: no per-block conditioning (use_film=False path).
    """
    def __init__(self, hidden_size: int, num_heads: int, cond_dim: int = 0,
                 expansion: float = 4.0, dropout: float = 0.0, cross_attn: bool = False):
        super().__init__()
        self.self_attn = SelfAttention(hidden_size, num_heads, dropout)
        self.mlp = SwiGLU(hidden_size, expansion, dropout)
        self.cross_attn = CrossAttention(hidden_size, num_heads) if cross_attn else None
        self.film = FiLM(cond_dim, hidden_size) if cond_dim > 0 else None

    def forward(self, x: torch.Tensor, film_cond: Optional[torch.Tensor] = None,
                cross_attn_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(rms_norm(x))
        if self.cross_attn is not None and cross_attn_ctx is not None:
            x = x + self.cross_attn(rms_norm(x), cross_attn_ctx)
        x = x + self.mlp(rms_norm(x))
        if self.film is not None and film_cond is not None:
            x = self.film(x, film_cond)
        return x


class TransformerLLevel(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int, cond_dim: int = 0,
                 expansion: float = 4.0, dropout: float = 0.0, cross_attn: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, cond_dim, expansion, dropout, cross_attn)
            for _ in range(num_layers)
        ])

    def forward(self, h: torch.Tensor, injection: torch.Tensor,
                film_cond: Optional[torch.Tensor] = None,
                cross_attn_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = h + injection
        for block in self.blocks:
            h = block(h, film_cond, cross_attn_ctx)
        return h


# =============================================================================
# UNet
# =============================================================================

class ResidualBlock(nn.Module):
    """Conv residual block with configurable conditioning mode.
    use_film=True:  FiLM on residual stream (non-skippable by construction)
    use_film=False: additive projection inside residual branch (original behavior)
    """
    def __init__(self, ch: int, cond_dim: int = 0, use_film: bool = True):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.use_film = use_film
        if cond_dim > 0:
            self.cond_mod = FiLM(cond_dim, ch) if use_film else nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, ch))
        else:
            self.cond_mod = None

    def forward(self, x: torch.Tensor, film_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if not self.use_film and self.cond_mod is not None and film_cond is not None:
            h = h + self.cond_mod(film_cond)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        x = x + h
        if self.use_film and self.cond_mod is not None and film_cond is not None:
            x = self.cond_mod(x, film_cond)
        return x


class UNetLLevel(nn.Module):
    def __init__(self, model_ch: int, num_blocks: int, cond_dim: int = 0, use_film: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(model_ch, cond_dim, use_film) for _ in range(num_blocks)
        ])

    def forward(self, h: torch.Tensor, injection: torch.Tensor,
                film_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = h + injection
        for block in self.blocks:
            h = block(h, film_cond)
        return h
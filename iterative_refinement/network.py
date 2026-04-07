import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class SwiGLU(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.0):
        super().__init__()
        inner = int(dim * expansion * 2 / 3)
        self.w1 = nn.Linear(dim, inner)
        self.w2 = nn.Linear(dim, inner)
        self.w3 = nn.Linear(inner, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dropout=dropout)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class PlanNet(nn.Module):
    """Per-layer plan generation with iterative refinement.
    Input: current qubit positions + gate structure + (optional) plan draft.
    Output: (batch, num_qubits, board_size) destination logits for all qubits.
    Non-relevant qubits are masked to stay in place.
    """
    def __init__(self, cfg):
        super().__init__()
        D = cfg.model.hidden_size
        self.board_size = cfg.board_size
        self.num_qubits = cfg.num_qubits
        # Encode qubit's current cell
        self.cell_embed = nn.Embedding(cfg.board_size, D)
        # Encode gate partner relationship
        self.gate_proj = nn.Linear(D, D, bias=False)
        self.has_gate_bias = nn.Parameter(torch.zeros(D))
        # Plan draft embedding: soft destination probs @ cell_embed.weight -> D
        self.plan_proj = nn.Linear(D, D, bias=False)
        # Learnable token for "selected for refinement"
        self.selected_emb = nn.Parameter(torch.zeros(D))
        # Qubit identity (sinusoidal, frozen)
        self.register_buffer('qubit_pe', _sinusoidal_pe(cfg.num_qubits, D))
        # Layer index embedding (which gate layer we're planning for)
        self.layer_embed = nn.Embedding(cfg.num_tasks, D)
        # Transformer backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(D, cfg.model.num_heads, cfg.model.dropout)
            for _ in range(cfg.model.num_blocks)
        ])
        self.out_norm = RMSNorm(D)
        self.out_proj = nn.Linear(D, cfg.board_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, src_cells, gate_partner_cells, layer_idx,
                plan_probs=None, selected=None):
        """
        Args:
            src_cells: (B, Q) flat cell index per qubit (current position)
            gate_partner_cells: (B, Q) flat cell of gate partner (-1 if no gate)
            layer_idx: (B,) which gate layer
            plan_probs: (B, Q, board_size) soft destination probs, None on first pass
            selected: (B, Q) bool mask of qubits being refined, None = all
        Returns:
            logits: (B, Q, board_size)
        """
        h = self.cell_embed(src_cells) + self.qubit_pe  # (B, Q, D)
        # Gate partner info
        has_gate = (gate_partner_cells >= 0)
        partner_emb = self.cell_embed(gate_partner_cells.clamp(min=0))
        h = h + has_gate.unsqueeze(-1).float() * (self.gate_proj(partner_emb) + self.has_gate_bias)
        # Layer context
        h = h + self.layer_embed(layer_idx)[:, None, :]
        # Plan draft injection (refinement passes)
        if plan_probs is not None:
            plan_emb = plan_probs @ self.cell_embed.weight  # (B, Q, D)
            h = h + self.plan_proj(plan_emb)
        # Selection signal
        if selected is not None:
            h = h + selected.unsqueeze(-1).float() * self.selected_emb
        for block in self.blocks:
            h = block(h)
        return self.out_proj(self.out_norm(h))


def _sinusoidal_pe(n, dim):
    pe = torch.zeros(n, dim)
    pos = torch.arange(n, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    if dim > 1:
        pe[:, 1::2] = torch.cos(pos * div[:dim // 2])
    return pe

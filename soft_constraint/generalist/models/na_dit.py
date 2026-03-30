"""NADiTModel: transformer + dual-stream iterative refinement for neutral atoms.

forward(init_cells, x_ans_soft, task_partner, z_H=None, z_L=None, selected=None)

z_H = [q_emb(N) | ans_emb(T*N)]  — question = initial positions, answer = plan.
z_L = reasoning scratchpad, carried across outer T steps.
injection = selected_emb at S positions in answer half — selection focus signal.
"""
import math
import torch
from torch import nn
from ml_collections import ConfigDict

from .trm_blocks import get_2d_sincos_pos_embed_rect, TransformerLLevel
from .ir_backbone import IRBackbone


class NADiTModel(nn.Module):
    def __init__(self, cfg: ConfigDict):
        super().__init__()
        d = cfg.hidden_size
        C = cfg.board_size           # vocab = H*W cell indices
        N = cfg.num_atoms
        T = cfg.num_gate_layers
        self.q_len = N               # question = initial atom positions
        self.ans_len = T * N         # answer = per-layer per-atom plan
        self.total_len = (T + 1) * N
        self.C = C

        # Token embedding: cell index -> hidden_size
        self.embed_tokens = nn.Embedding(C, d)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        self.scale = math.sqrt(d)

        # Position embedding: 2D sincos on (T+1, N) grid (layer x atom)
        pos = get_2d_sincos_pos_embed_rect(d, T + 1, N)   # ((T+1)*N, d)
        self.register_buffer('pos_embed', pos)

        # Task partner embedding: added to answer tokens to encode gate structure
        self.partner_emb = nn.Embedding(N + 1, d)  # index N = "no gate" sentinel
        nn.init.normal_(self.partner_emb.weight, std=0.02)

        # IR backbone
        l_level = TransformerLLevel(
            d, cfg.num_heads, cfg.num_blocks,
            0, cfg.expansion, cfg.dropout, False)
        self.ir = IRBackbone(l_level, cfg.H_cycle, cfg.n)

        # Selection focus signal
        self.selected_emb = nn.Parameter(torch.zeros(d))

        # Output projection
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, C)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, init_cells, x_ans_soft, task_partner,
                z_H=None, z_L=None, selected=None):
        """
        Args:
            init_cells:   (B, N) int — initial atom cell indices
            x_ans_soft:   (B, T*N, C) float — soft plan distributions
            task_partner: (B, T*N) int — partner atom index per answer token (N = no gate)
            z_H:          carried latent (or None for first call)
            z_L:          carried scratchpad (or None for first call)
            selected:     (B, T*N) bool — which answer tokens to focus on
        Returns:
            logits: (B, T*N, C)
            z_H:    detached
            z_L:    detached
        """
        if z_H is None:
            # Question: embed initial positions
            q_emb = self.embed_tokens(init_cells) * self.scale          # (B, N, d)
            q_emb = q_emb + self.pos_embed[:self.q_len]

            # Answer: soft embedding + task partner
            ans_emb = (x_ans_soft @ self.embed_tokens.weight) * self.scale  # (B, T*N, d)
            ans_emb = ans_emb + self.pos_embed[self.q_len:]
            ans_emb = ans_emb + self.partner_emb(task_partner)

            z_H = torch.cat([q_emb, ans_emb], dim=1)                   # (B, (T+1)*N, d)
        z_L = z_L if z_L is not None else torch.zeros_like(z_H)

        # Injection: selected_emb at answer positions
        injection = torch.zeros_like(z_H)
        injection[:, self.q_len:] = selected.unsqueeze(-1).float() * self.selected_emb

        z_H, z_L = self.ir(z_H, z_L, injection)
        logits = self.out_proj(self.out_norm(z_H[:, self.q_len:]))      # (B, T*N, C)
        return logits, z_H.detach(), z_L.detach()

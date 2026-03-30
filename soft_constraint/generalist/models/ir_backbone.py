"""Dual-stream iterative refinement backbone.

z_H: answer draft — holds [question | answer], read out at the end.
z_L: reasoning scratchpad — sees everything via injection, never read out.

Per H_cycle:
    for _ in range(n):
        z_L = l_level(z_L, z_H + injection)  # z_L sees question + answer + x_t
    z_H = l_level(z_H, z_L)                  # z_H sees only z_L's summary

Degeneracy:
    H_cycle=1, n=0 → vanilla single pass (z_L inert, one l_level call)
    H_cycle=1, n=1 → minimal TRM cycle (2 l_level calls)
    H_cycle>1       → gradient truncation on first H_cycle-1 cycles
"""
from typing import Tuple
import torch
from torch import nn


class IRBackbone(nn.Module):
    def __init__(self, l_level: nn.Module, H_cycle: int = 1, n: int = 1):
        super().__init__()
        self.l_level = l_level
        self.H_cycle = H_cycle
        self.n = n

    def forward(self, z_H: torch.Tensor, z_L: torch.Tensor,
                injection: torch.Tensor, **l_level_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        def h_cycle(z_H, z_L):
            for _ in range(self.n):
                z_L = self.l_level(z_L, z_H + injection, **l_level_kwargs)
            z_H = self.l_level(z_H, z_L, **l_level_kwargs)
            return z_H, z_L

        with torch.no_grad():
            for _ in range(self.H_cycle - 1):
                z_H, z_L = h_cycle(z_H, z_L)
        return h_cycle(z_H, z_L)
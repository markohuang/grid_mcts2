"""Config for TRM-DiT model (neutral atoms).

Effective depth per forward = H_cycle * (n+1) * num_blocks
T = number of partial-update refinement steps per training step.
"""
from ml_collections import ConfigDict

VARIANTS = {
    # Tiny — fast smoke testing
    'tiny_T4N2': dict(T=4, n=2, num_blocks=1, hidden_size=96, num_heads=4),
    # ~1M params (iter-level)
    'iter_T4N4': dict(T=4, n=4, num_blocks=2, hidden_size=144, num_heads=4),
    'iter_T8N4': dict(T=8, n=4, num_blocks=2, hidden_size=144, num_heads=4),
    'iter_T4N8': dict(T=4, n=8, num_blocks=2, hidden_size=144, num_heads=4),
    'iter_T4H2N4': dict(T=4, H_cycle=2, n=4, num_blocks=2, hidden_size=144, num_heads=4),
    # ~2.3M params (base-level)
    'base_T4N2': dict(T=4, n=2, num_blocks=3, hidden_size=192, num_heads=4),
    'base_T4N4': dict(T=4, n=4, num_blocks=3, hidden_size=192, num_heads=4),
    'base_T8N4': dict(T=8, n=4, num_blocks=3, hidden_size=192, num_heads=4),
}


def get_config(variant='iter_T4N4'):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown TRM-DiT variant: {variant}. Choose from {list(VARIANTS.keys())}")
    v = VARIANTS[variant]
    cfg = ConfigDict()

    cfg.hidden_size = v['hidden_size']
    cfg.num_heads = v['num_heads']
    cfg.num_blocks = v['num_blocks']
    cfg.expansion = v.get('expansion', 4.0)
    cfg.dropout = 0.0

    cfg.T = v['T']
    cfg.H_cycle = v.get('H_cycle', 1)
    cfg.n = v.get('n', 2)

    # board_size, num_atoms, num_gate_layers injected by experiment
    return cfg

"""HPC self-play preset. Inherits config.py and overrides scale knobs.

Selected via `--preset=hpc` in selfplay_worker.py (and future train_offline.py).
CLI flags still override everything here, so smoke tests can dial num_simulations
back down via e.g. --config.mcts.num_simulations=50 on the HPC preset.

Scope is intentionally narrow: only knobs that should differ between local dev
and HPC self-play. Training-side knobs (batch_size, buffer_size, etc.) are not
touched here because train_offline.py is not yet implemented.
"""

from .config import get_config as _base_get_config


def get_config():
    c = _base_get_config()
    # AlphaDev=800, AlphaZero=800, lc0 training=800. Matches the 800-sim target
    # that pb_c_base=500 (in config.py) is tuned for. Scale further later.
    c.mcts.num_simulations = 800
    return c

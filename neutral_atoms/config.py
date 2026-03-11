import collections
from dataclasses import dataclass, field
from typing import Callable, Optional

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


@dataclass
class MCTSConfig:
    num_simulations: int = 50
    discount: float = 1.0
    pb_c_base: int = 19652
    pb_c_init: float = 1.25
    root_dirichlet_alpha: float = 0.03
    root_exploration_fraction: float = 0.25
    known_bounds: KnownBounds = field(default_factory=lambda: KnownBounds(-6.0, 6.0))
    max_moves: float = float('inf')
    visit_softmax_temperature_fn: Optional[Callable] = None

    def __post_init__(self):
        if self.visit_softmax_temperature_fn is None:
            self.visit_softmax_temperature_fn = lambda steps: (
                2.0 if steps < 500e3 else 0.5 if steps < 750e3 else 0.25
            )


@dataclass
class TrainingConfig:
    epochs: int = 50
    num_selfplay: int = 20
    buffer_size: int = 1000
    td_steps: int = 5
    batch_size: int = 128
    lr: float = 2e-4
    training_steps: int = 200
    grad_norm_clip: float = 1.0
    log_interval: int = 200
    save_dir: str = './checkpoints/'
    accelerator: str = 'cpu'
    devices: int = 1
    seed: int = 12315


@dataclass
class NetworkConfig:
    num_tasks: int = 3
    num_qubits: int = 9
    board_size: int = 12
    num_actions: int = 109
    v_hsize: int = 64
    p_hsize: int = 32
    mlp_depth: int = 2
    ema_decay: float = 0.995
    num_bins: int = 51
    value_min: float = -25.0
    value_max: float = 25.0

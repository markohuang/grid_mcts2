"""Single fixed map experiment for overfitting validation."""
from ml_collections import ConfigDict


SINGLE_MAP = {
    'board_dim': (5, 5), 'num_qubits': 12,
    'atom_map': [4, 12, 7, 11, 15, 5, 2, 21, 22, 23, 20, 8],
    'tasks': [
        [[4, 3], [6, 7], [9, 8], [10, 11]],
        [[2, 1], [4, 5], [7, 6], [8, 9]],
        [[0, 1], [2, 3], [6, 5], [9, 8]],
    ],
}


def apply_single_map_config(cfg):
    cfg.experiment = 'single_map'

    cfg.env = ConfigDict()
    cfg.env.board_height = 5
    cfg.env.board_width = 5
    cfg.env.num_qubits = 12
    cfg.env.num_layers = 3
    cfg.env.gates_per_layer = 4

    cfg.training = ConfigDict()
    cfg.training.select_prob = 0.3

    cfg.surrogate = ConfigDict()
    cfg.surrogate.lambda_g = 1.0
    cfg.surrogate.lambda_r = 1.0
    cfg.surrogate.lambda_aux = 0.0
    cfg.surrogate.loss_mode = 'huber'
    cfg.surrogate.delta = 0.1

    cfg.data = ConfigDict()
    cfg.data.task_id = 'single_map'
    cfg.data.batch_size = 32
    cfg.data.num_workers = 0

    cfg.data.augmentation = ConfigDict()
    cfg.data.augmentation.enabled = True
    cfg.data.augmentation.transforms = ('atom_permutation', 'rotation', 'reflection', 'transpose')

    return cfg


def inject_shared_properties(cfg):
    C = cfg.env.board_height * cfg.env.board_width
    cfg.model.board_size = C
    cfg.model.num_atoms = cfg.env.num_qubits
    cfg.model.num_gate_layers = cfg.env.num_layers

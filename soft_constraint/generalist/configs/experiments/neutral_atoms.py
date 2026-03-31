"""Experiment config for neutral atoms reconfiguration."""
from ml_collections import ConfigDict


def apply_neutral_atoms_config(cfg):
    cfg.experiment = 'neutral_atoms'

    # Environment
    cfg.env = ConfigDict()
    cfg.env.board_height = 5
    cfg.env.board_width = 5
    cfg.env.num_qubits = 12
    cfg.env.num_layers = 3
    cfg.env.gates_per_layer = 4

    # Training
    cfg.training = ConfigDict()
    cfg.training.select_prob = 0.3
    cfg.training.regurgitate_steps = 3  # feed output back as input N times per batch

    # Surrogate loss weights
    cfg.surrogate = ConfigDict()
    cfg.surrogate.lambda_g = 1.0
    cfg.surrogate.lambda_r = 1.0
    cfg.surrogate.lambda_aux = 0.0
    cfg.surrogate.loss_mode = 'huber'
    cfg.surrogate.delta = 0.1

    # Data
    cfg.data = ConfigDict()
    cfg.data.task_id = 'neutral_atoms'
    cfg.data.batch_size = 64
    cfg.data.num_workers = 8
    cfg.data.val_pool_size = 500
    cfg.data.val_seed = 99999

    cfg.data.augmentation = ConfigDict()
    cfg.data.augmentation.enabled = True
    cfg.data.augmentation.transforms = ('atom_permutation', 'rotation', 'reflection', 'transpose')

    return cfg


def inject_shared_properties(cfg):
    C = cfg.env.board_height * cfg.env.board_width
    cfg.model.board_size = C
    cfg.model.num_atoms = cfg.env.num_qubits
    cfg.model.num_gate_layers = cfg.env.num_layers

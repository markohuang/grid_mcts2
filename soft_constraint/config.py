import random
import ml_collections


# ─────────────────────────────────────────────────────────────────────────────
# Map generator (standalone, mirrors neutral_atoms/map_generator.py)
# ─────────────────────────────────────────────────────────────────────────────

def _random_matching(num_qubits, num_pairs, rng):
    qubits = list(range(num_qubits))
    rng.shuffle(qubits)
    pairs = []
    for i in range(0, min(2 * num_pairs, len(qubits)) - 1, 2):
        pairs.append([qubits[i], qubits[i + 1]])
    return pairs


def generate_random_map(board_dim, num_qubits, num_layers,
                        gates_per_layer=None, seed=None):
    rng = random.Random(seed)
    board_h, board_w = board_dim
    board_size = board_h * board_w
    assert num_qubits <= board_size
    if gates_per_layer is None:
        gates_per_layer = num_qubits // 2
    all_cells = list(range(board_size))
    rng.shuffle(all_cells)
    atom_map = all_cells[:num_qubits]
    tasks = [_random_matching(num_qubits, gates_per_layer, rng)
             for _ in range(num_layers)]
    return {
        'board_dim': board_dim,
        'num_qubits': num_qubits,
        'atom_map': atom_map,
        'tasks': tasks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Map definitions
# ─────────────────────────────────────────────────────────────────────────────

MAPS = {
    0: {  # 2x6 with 9 atoms
        'board_dim': (2, 6), 'num_qubits': 9,
        'atom_map': [11, 3, 1, 8, 10, 7, 5, 6, 4],
        'tasks': [
            [[0, 8], [6, 2], [1, 7], [3, 5]],
            [[0, 2], [6, 8], [1, 5], [3, 7]],
            [[3, 7], [1, 5], [6, 8]],
        ],
    },
    1: {  # 4x4 with 8 atoms
        'board_dim': (4, 4), 'num_qubits': 8,
        'atom_map': [5, 9, 10, 11, 2, 6, 0, 14],
        'tasks': [
            [[0, 1], [2, 3], [5, 4], [6, 7]],
            [[0, 1], [2, 3], [5, 4], [6, 7]],
            [[0, 1], [2, 3], [5, 4], [6, 7]],
        ],
    },
    2: {  # 5x5 with 12 atoms
        'board_dim': (5, 5), 'num_qubits': 12,
        'atom_map': [4, 12, 7, 11, 15, 5, 2, 21, 22, 23, 20, 8],
        'tasks': [
            [[4, 3], [6, 7], [9, 8], [10, 11]],
            [[2, 1], [4, 5], [7, 6], [8, 9]],
            [[0, 1], [2, 3], [6, 5], [9, 8]],
        ],
    },
    3: {  # 8x8 with 20 atoms (generated)
        'board_dim': (8, 8), 'num_qubits': 20,
        '_generator': {'seed': 42, 'num_layers': 5, 'gates_per_layer': 10},
    },
    4: {  # 8x8 with 20 atoms, different seed
        'board_dim': (8, 8), 'num_qubits': 20,
        '_generator': {'seed': 123, 'num_layers': 5, 'gates_per_layer': 10},
    },
    5: {  # 8x8 with 20 atoms, different seed
        'board_dim': (8, 8), 'num_qubits': 20,
        '_generator': {'seed': 999, 'num_layers': 5, 'gates_per_layer': 10},
    },
}


def _resolve_map(m):
    if 'atom_map' in m:
        return m
    gen = m['_generator']
    return generate_random_map(
        m['board_dim'], m['num_qubits'],
        gen['num_layers'], gen.get('gates_per_layer'), gen['seed'],
    )


def atom_map_to_positions(atom_map, board_width):
    return [(idx // board_width, idx % board_width) for idx in atom_map]


def get_config():
    c = ml_collections.ConfigDict()
    c.map_num = 2

    c.env = ml_collections.ConfigDict()
    # board dims filled by set_derived_config

    c.optimizer = ml_collections.ConfigDict()
    c.optimizer.lr = 0.03
    c.optimizer.n_steps = 400
    c.optimizer.n_restarts = 10
    c.optimizer.logit_init_scale = 0.3
    c.optimizer.logit_bias = 4.0

    c.surrogate = ml_collections.ConfigDict()
    c.surrogate.lambda_g = 1.0
    c.surrogate.lambda_r = 1.0
    c.surrogate.lambda_aux = 0.0
    c.surrogate.loss_mode = 'huber'  # 'log', 'linear', or 'huber'
    c.surrogate.delta = 0.1          # threshold for huber mode (max 1/δ gradient amplification)

    c.experiment = ml_collections.ConfigDict()
    c.experiment.output_dir = './outputs'
    c.experiment.eval_interval = 100   # evaluate true cost every N steps
    c.experiment.seed = 42

    c.training = ml_collections.ConfigDict()
    c.training.accelerator = 'auto'
    c.training.devices = 1

    return c


def set_derived_config(config):
    m = _resolve_map(MAPS[config.map_num])
    H, W = m['board_dim']
    with config.unlocked():
        config.env.board_height = H
        config.env.board_width = W
        config.env.board_size = H * W
        config.env.num_qubits = m['num_qubits']
        config.env.num_layers = len(m['tasks'])


def get_map_data(config):
    return _resolve_map(MAPS[config.map_num])

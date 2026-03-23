import ml_collections


MAPS = [
    {   # 2x6 with 9 atoms
        'board_dim': (2, 6), 'num_qubits': 9,
        'atom_map': [11, 3, 1, 8, 10, 7, 5, 6, 4],
        'tasks': [
            [[0, 8], [6, 2], [1, 7], [3, 5]],
            [[0, 2], [6, 8], [1, 5], [3, 7]],
            [[3, 7], [1, 5], [6, 8]],
        ],
    },
    {   # 4x4 with 8 atoms
        'board_dim': (4, 4), 'num_qubits': 8,
        'atom_map': [5, 9, 10, 11, 2, 6, 0, 14],
        'tasks': [
            [[0, 1], [2, 3], [5, 4], [6, 7]],
            [[0, 1], [2, 3], [5, 4], [6, 7]],
            [[0, 1], [2, 3], [5, 4], [6, 7]],
        ],
    },
    {   # 5x5 with 12 atoms
        'board_dim': (5, 5), 'num_qubits': 12,
        'atom_map': [4, 12, 7, 11, 15, 5, 2, 21, 22, 23, 20, 8],
        'tasks': [
            [[4, 3], [6, 7], [9, 8], [10, 11]],
            [[2, 1], [4, 5], [7, 6], [8, 9]],
            [[0, 1], [2, 3], [6, 5], [9, 8]],
        ],
    },
    {   # 8x8 with 20 atoms (fixed seed for reproducibility)
        'board_dim': (8, 8), 'num_qubits': 20,
        'atom_map': None,  # generated via map_generator
        'tasks': None,
        '_generator': {'seed': 42, 'num_layers': 5, 'gates_per_layer': 10},
    },
    {   # 8x8 with 30 atoms
        'board_dim': (8, 8), 'num_qubits': 30,
        'atom_map': None,
        'tasks': None,
        '_generator': {'seed': 42, 'num_layers': 5, 'gates_per_layer': 15},
    },
]


def atom_map_to_positions(atom_map, board_width):
    return [(idx // board_width, idx % board_width) for idx in atom_map]


def get_config():
    c = ml_collections.ConfigDict()
    c.map_num = 0
    c.use_fake = False
    c.random_board = False  # if True, generate random board each game
    c.random_board_seed = -1  # -1 = different seed each game

    c.env = ml_collections.ConfigDict()
    c.env.reward_scale = 1.0
    c.env.reward_mode = 'plan_cost'  # 'plan_cost', 'layer_delta', 'layer_completion', 'remaining_cost'
    c.env.entropy_weight = 0.0  # tie-breaker: adds entropy_weight * within_group_entropy to cost (0 = off)

    c.mcts = ml_collections.ConfigDict()
    c.mcts.num_simulations = 50
    c.mcts.discount = 1.0
    c.mcts.pb_c_base = 19652
    c.mcts.pb_c_init = 1.25
    c.mcts.root_dirichlet_alpha = 0.03
    c.mcts.root_exploration_fraction = 0.25
    c.mcts.known_bounds = ml_collections.ConfigDict({'min': -6.0, 'max': 6.0})
    c.mcts.max_moves = 10000
    c.mcts.temperature_init = 2.0
    c.mcts.temperature_final = 0.25
    c.mcts.temperature_decay_steps = 1000

    c.training = ml_collections.ConfigDict()
    c.training.epochs = 50
    c.training.num_selfplay = 20
    c.training.buffer_size = 50000
    c.training.td_steps = 5  # for layer_completion reward, set >= max_atoms_per_layer (see set_derived_config)
    c.training.batch_size = 128
    c.training.lr = 2e-4
    c.training.training_steps = 200
    c.training.grad_norm_clip = 1.0
    c.training.log_interval = 200
    c.training.accelerator = 'auto'
    c.training.devices = 1
    c.training.seed = 12315
    c.training.policy_target_temperature = 1.0
    c.training.num_parallel_games = 1
    c.training.priority_exponent = 0.0  # 0 = uniform sampling, >0 = prioritize low-cost games
    c.training.data_augmentation = False  # apply random board symmetries to training batches

    c.experiment = ml_collections.ConfigDict()
    c.experiment.output_dir = './outputs'
    c.experiment.checkpoint_every_n_epochs = 10
    c.experiment.early_stopping_patience = 10
    c.experiment.load_checkpoint = ''       # path to load model weights from before training
    c.experiment.curriculum_maps = 0        # number of random maps to add progressively (0 = off)
    c.experiment.curriculum_patience = 20   # epochs without improvement before adding next map
    c.experiment.curriculum_initial_phase = 0  # pre-populate map pool for resuming mid-curriculum
    c.training.fixed_map_fraction = 0.0    # fraction of selfplay games using the fixed eval map (0 = off)

    c.network = ml_collections.ConfigDict()
    c.network.v_hsize = 64
    c.network.p_hsize = 32
    c.network.mlp_depth = 2
    c.network.ema_decay = 0.995
    c.network.num_bins = 101
    c.network.value_min = -20.0
    c.network.value_max = 5.0
    c.network.correctness_weight = 1.0
    c.network.latency_weight = 0.1

    return c


def _resolve_map(m):
    if m.get('atom_map') is not None:
        return m
    from .map_generator import generate_random_map
    gen = m['_generator']
    generated = generate_random_map(
        m['board_dim'], m['num_qubits'], gen['num_layers'],
        gates_per_layer=gen.get('gates_per_layer'), seed=gen['seed'],
    )
    return generated


def set_derived_config(config):
    m = _resolve_map(MAPS[config.map_num])
    board_h, board_w = m['board_dim']
    num_qubits = m['num_qubits']
    board_size = board_h * board_w
    with config.unlocked():
        config.env.board_height = board_h
        config.env.board_width = board_w
        config.env.num_qubits = num_qubits
        config.network.num_tasks = len(m['tasks'])
        config.network.num_qubits = num_qubits
        config.network.board_size = board_size
        config.network.num_actions = board_size
        # max atoms per layer = unique qubits in the largest layer (used for td_steps guidance)
        tasks = m['tasks']
        config.env.max_atoms_per_layer = max(
            len({q for pair in layer for q in pair}) for layer in tasks
        )


def get_map_data(config):
    m = _resolve_map(MAPS[config.map_num])
    return m

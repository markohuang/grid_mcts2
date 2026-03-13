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
]


def atom_map_to_positions(atom_map, board_width):
    return [(idx // board_width, idx % board_width) for idx in atom_map]


def get_config():
    c = ml_collections.ConfigDict()
    c.map_num = 0
    c.use_fake = False

    c.env = ml_collections.ConfigDict()
    c.env.budget = 24
    c.env.entropy_weight = 0.0
    c.env.reward_scale = 1.0
    c.env.reward_mode = 'cost_delta'

    c.mcts = ml_collections.ConfigDict()
    c.mcts.num_simulations = 50
    c.mcts.discount = 1.0
    c.mcts.pb_c_base = 19652
    c.mcts.pb_c_init = 1.25
    c.mcts.root_dirichlet_alpha = 0.03
    c.mcts.root_exploration_fraction = 0.25
    c.mcts.known_bounds = ml_collections.ConfigDict({'min': -25.0, 'max': 25.0})
    c.mcts.max_moves = 10000

    c.training = ml_collections.ConfigDict()
    c.training.epochs = 50
    c.training.num_selfplay = 20
    c.training.buffer_size = 1000
    c.training.td_steps = 5
    c.training.batch_size = 128
    c.training.lr = 2e-4
    c.training.training_steps = 200
    c.training.grad_norm_clip = 1.0
    c.training.log_interval = 200
    c.training.accelerator = 'auto'
    c.training.devices = 1
    c.training.seed = 12315
    c.training.policy_entropy_weight = 0.0
    c.training.policy_target_temperature = 1.0
    c.training.num_parallel_games = 1
    c.training.pretrain_value_steps = 0
    c.training.aux_value_weight = 0.0
    c.training.aux_value_samples = 512
    c.training.freeze_value_epochs = 0

    c.experiment = ml_collections.ConfigDict()
    c.experiment.output_dir = './outputs'
    c.experiment.checkpoint_every_n_epochs = 10
    c.experiment.early_stopping_patience = 10

    c.network = ml_collections.ConfigDict()
    c.network.v_hsize = 64
    c.network.p_hsize = 32
    c.network.mlp_depth = 2
    c.network.ema_decay = 0.995
    c.network.num_bins = 101
    c.network.value_min = -25.0
    c.network.value_max = 25.0
    c.network.correctness_weight = 1.0
    c.network.latency_weight = 1.0

    return c


def set_derived_config(config):
    m = MAPS[config.map_num]
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
        config.network.num_actions = 1 + num_qubits * board_size

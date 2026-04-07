import ml_collections
from neutral_atoms.config import MAPS, _resolve_map


def get_config():
    cfg = ml_collections.ConfigDict()
    cfg.map_num = 3
    cfg.random_board = True
    cfg.random_board_seed = -1  # -1 = different seed each sample

    cfg.model = ml_collections.ConfigDict()
    cfg.model.hidden_size = 128
    cfg.model.num_heads = 4
    cfg.model.num_blocks = 4
    cfg.model.dropout = 0.0
    cfg.model.T = 4               # refinement passes (ACT supervision loop)
    cfg.model.select_prob = 0.3   # fraction of qubits re-sampled per refinement step

    cfg.training = ml_collections.ConfigDict()
    cfg.training.batch_size = 64
    cfg.training.lr = 3e-4
    cfg.training.weight_decay = 1e-4
    cfg.training.max_steps = 50000
    cfg.training.warmup_steps = 1000
    cfg.training.grad_clip = 1.0
    cfg.training.top_k = 3
    cfg.training.collision_weight = 10.0
    cfg.training.reconfig_weight = 1.0
    cfg.training.seed = 42
    cfg.training.log_every = 100
    cfg.training.eval_every = 1000
    cfg.training.save_every = 5000
    cfg.training.num_eval_samples = 200

    cfg.trainer = ml_collections.ConfigDict()
    cfg.trainer.accelerator = 'auto'
    cfg.trainer.devices = 1
    cfg.trainer.precision = 'bf16-mixed'

    cfg.experiment = ml_collections.ConfigDict()
    cfg.experiment.output_dir = './outputs_ir'

    return cfg


def set_derived_config(cfg):
    m = _resolve_map(MAPS[cfg.map_num])
    board_h, board_w = m['board_dim']
    with cfg.unlocked():
        cfg.board_height = board_h
        cfg.board_width = board_w
        cfg.board_size = board_h * board_w
        cfg.num_qubits = m['num_qubits']
        cfg.num_tasks = len(m['tasks'])
        cfg.gates_per_layer = len(m['tasks'][0])
    return m

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
    {   # MAPS[5]: 8x8 with 20 atoms, 5 layers, 6 gates/layer (sparser than MAPS[3]).
        # Used as the v4 structure template for random_board=True (fresh map per game).
        'board_dim': (8, 8), 'num_qubits': 20,
        'atom_map': None,
        'tasks': None,
        '_generator': {'seed': 42, 'num_layers': 5, 'gates_per_layer': 6},
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
    # Device for network inference inside selfplay workers ('cpu' or 'cuda').
    # 'cpu' is always safe (default). 'cuda' moves the real net to GPU inside each
    # subprocess — use with num_workers small enough to fit N CUDA contexts in VRAM
    # (each context ~200-400 MB overhead beyond model weights; 4 workers is safe on A100).
    c.selfplay_device = 'cpu'

    c.env = ml_collections.ConfigDict()
    c.env.reward_scale = 1.0
    # Reward mode. Confirmed winner (2026-04-21): plan_cost.
    #   v2j A/B on fixed map 2 (10k sims, FakeNet, AlphaDev defaults):
    #     plan_cost  -> cost min=12 (= Round-04 best), median=21
    #     layer_delta -> cost min=13, median=23
    #   v3a A/B on random 5x5 maps (same config):
    #     plan_cost  -> cost min=12, median=22
    #     layer_delta -> cost min=14, median=24
    #   Two independent wins on cost distribution AND correlations; plan_cost also reaches
    #   terminal more often (0.33 vs 0.25), has stronger corr(depth,cost) (-0.30 vs -0.12),
    #   and runs ~32% faster per game (more sims skip NN inference at terminal leaves).
    #
    # Alternatives to test (especially once value head is trained):
    #   'layer_delta'      - per-layer cost deltas. Gives broader policy targets (entropy 1.84 vs
    #                        plan_cost's 1.28). May be preferable for cold-start training where the
    #                        NN needs more policy-distribution signal per state. Not tested with
    #                        trained value head yet — may outperform plan_cost there.
    #   'layer_completion' - sparse reward at layer boundaries only. Never formally A/B-tested.
    #   'remaining_cost'   - -cost/episode_length each step (dense reward envelope). Never A/B-tested.
    #
    # Also: `c.mcts.prior_mix_weight` (see below) is a complementary heuristic that only works
    # with reward_mode='layer_delta' (fail-fast assertion against plan_cost + prior_mix).
    c.env.reward_mode = 'plan_cost'
    c.env.entropy_weight = 0.0  # tie-breaker: adds entropy_weight * within_group_entropy to cost (0 = off)
    c.env.track_plan_delta = False  # expose plan_cost delta in step info for diagnostics / MCTS heuristics

    c.mcts = ml_collections.ConfigDict()
    # num_simulations: smoke-test default. HPC preset overrides to 800 (AlphaDev parity).
    # See config_hpc.py. Override via --config.mcts.num_simulations=<N> on the CLI.
    c.mcts.num_simulations = 50
    c.mcts.discount = 1.0
    # pb_c = init + log((N_parent + base + 1) / base). At N_parent = base, log term adds
    # log(2) ~= 0.69 on top of init.
    #   AlphaDev / AlphaZero: base=19652, init=1.25   (what we use)
    #   lc0 training:         base=38739, init=1.745  (tuned for 10k-100k tournament sims)
    # Growth at base=19652:
    #   N_parent=800   -> pb_c = 1.29 (+3% over init)
    #   N_parent=10000 -> pb_c = 1.66 (+33%)
    #   N_parent=50000 -> pb_c = 2.63 (+110%)
    # An earlier session tried base=500 (+244% at 10k sims) on the theory that the log term
    # should "engage at our scale". v2e (base=500) vs v2f (base=19652) A/B with 10k sims +
    # FakeNet + layer_delta showed 19652 wins across the board: cost median 28->23, cost min
    # 16->13 (Round-04 optimum was 12), trajectory diversity 17.8%->17.8% (same), AND first
    # time corr(depth,cost) flipped negative. AlphaDev's default was right.
    c.mcts.pb_c_base = 19652  # AlphaDev/Round-04 default (see experiments/selfplay_v2f)
    c.mcts.pb_c_init = 1.25   # AlphaDev/AlphaZero; lc0=1.745 if we want more uniform exploration
    # root_dirichlet_alpha: AlphaDev=0.03, AlphaZero chess=0.03, Go/shogi=0.15. Our branching
    # factor is 9-15. v2e-vs-v2f A/B showed 0.3 (very diffuse) regresses cost vs 0.03 (spiky).
    # v2f-vs-v2h A/B (2026-04-20) showed 0.1 matches 0.03 on cost with slightly stronger
    # correlations and larger top-K structural gaps -- adopted as default going into v3.
    # Intuition: alpha=0.03 is near-one-hot noise (single action boost per game); alpha=0.1
    # spreads noise across 2-4 actions per game, giving more within-game action coverage
    # without washing out the Q signal the way 0.3 did.
    c.mcts.root_dirichlet_alpha = 0.1  # v2h default (alpha=0.1); see docs/selfplay/selfplay_waves.md
    c.mcts.root_exploration_fraction = 0.25  # AlphaDev=0.25, AlphaZero=0.25
    c.mcts.known_bounds = ml_collections.ConfigDict({'min': -6.0, 'max': 6.0})
    c.mcts.max_moves = 10000
    # Temperature: lc0-style per-episode linear decay (see docs/selfplay/lc0_temperature_design.md).
    # get_temperature(move_in_episode, config) linearly interpolates from temperature_init to
    # temperature_final over temperature_decay_moves moves, then stays at temperature_final.
    # Setting temperature_decay_moves=0 degenerates to a constant T = temperature_init (current default,
    # matches v2d/v2e/v2f behavior where training_steps=0 made the old schedule also effectively constant).
    #
    # To activate per-move decay later (lc0-style): try temperature_init=1.0, temperature_final=0.0,
    # temperature_decay_moves=8 for 24-step episodes -> moves 0-7 decay 1.0->0, moves 8-23 argmax.
    # Each game exposes the full E/E spectrum on its own.
    c.mcts.temperature_init = 1.0
    c.mcts.temperature_final = 1.0  # unused when decay_moves=0
    c.mcts.temperature_decay_moves = 0  # 0 = constant T throughout every game (= current behavior)
    # Plan-cost heuristic mixed into MCTS prior. When > 0, _expand_node probes each legal
    # action one step on a cloned env and folds β · plan_cost_delta / cost_ub into the
    # log-prior before softmax. The heuristic then rides the usual pb_c · √(ΣN)/(1+N)
    # envelope and distills into π_θ via visit counts. Only valid with reward_mode =
    # 'layer_delta' (plan_cost mode already puts the same delta into the reward/value
    # target; stacking both would double-count). Auto-enables env.track_plan_delta.
    c.mcts.prior_mix_weight = 0.0
    # fast_mcts backend knobs (read by fast_mcts/search.py; classic backend ignores them).
    # backend='classic' → neutral_atoms.mcts.play_game (default, unchanged behaviour).
    # backend='fast'    → fast_mcts.search.play_game (leaf-gather + virtual loss, Phase 1).
    # nn_batch_size and virtual_loss only take effect when backend='fast'.
    # Pairing rationale: without VL, 32 leaves descend greedily and mostly collide on the
    # same unexpanded node (UCB doesn't update until backup). VL=1.0 forces them onto
    # different branches, so batch=32 actually fills the batch. bench shows 4.8× at this
    # pairing; 5.35× at batch=32/vl=1.5. Classic is sequential and never uses VL at all.
    c.mcts.backend = 'classic'
    c.mcts.nn_batch_size = 32   # leaves gathered per NN call (fast backend only)
    c.mcts.virtual_loss = 1.0   # subtracted from value_sum on in-flight paths (fast backend only)

    # Gumbel AlphaZero (docs/gumbel_pczero_plan.md). When enabled, replaces root Dirichlet +
    # pUCT with Gumbel-Top-m + sequential halving, and non-root pUCT with the deterministic
    # improved-policy rule (argmax_a [π'(a) - N(a)/(1+ΣN)]). Policy target becomes π' (guaranteed
    # improvement) instead of softmax-of-visits.
    #   num_samples_m=0 => auto-set to num_legal = board_size - num_qubits + 1 in set_derived_config.
    #   c_visit / c_scale: σ(q_norm) = (c_visit + max_N) * c_scale * q_norm. Paper defaults.
    c.mcts.gumbel = ml_collections.ConfigDict()
    c.mcts.gumbel.enabled = False
    c.mcts.gumbel.num_samples_m = 0
    c.mcts.gumbel.c_visit = 50.0
    c.mcts.gumbel.c_scale = 1.0

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
    c.experiment.study = ''                 # grouping key: e.g. round08_reward_signal
    c.experiment.hypothesis = ''            # short id: e.g. H1
    c.experiment.variant = ''               # control/treatment/run label
    c.experiment.tags = ''                  # comma-separated labels for filtering
    c.experiment.notes = ''                 # concise human note saved with the run
    c.experiment.parent_run = ''            # prior run/checkpoint this run builds on
    c.experiment.decision = ''              # optional intended decision if hypothesis is confirmed
    c.training.fixed_map_fraction = 0.0    # fraction of selfplay games using the fixed eval map (0 = off)

    c.network = ml_collections.ConfigDict()
    c.network.v_hsize = 64
    # 2026-04-20: bumped 32 -> 64 to match v_hsize. At dim=32 the policy transformer has
    # nhead=2 / 16 dims-per-head (R11 verdict): too narrow for meaningful attention and
    # caused 5x slower specialist convergence.
    c.network.p_hsize = 64
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
        # Gumbel num_samples_m defaults to legal-action count (constant per map:
        # board_size - num_qubits + 1 since every step has num_qubits occupied cells,
        # and legal = [current_cell] + [empty cells]). See board.py:get_legal_actions_for_qubit.
        if config.mcts.gumbel.num_samples_m == 0:
            config.mcts.gumbel.num_samples_m = board_size - num_qubits + 1


def get_map_data(config):
    m = _resolve_map(MAPS[config.map_num])
    return m


def map_class(map_data):
    h, w = map_data['board_dim']
    q = map_data['num_qubits']
    tasks = map_data['tasks']
    g = max(len(layer) for layer in tasks)
    l = len(tasks)
    return f'{h}x{w}_{q}qb_{g}gpl_{l}lyrs'


def map_id(map_data):
    import hashlib, json
    canonical = json.dumps({
        'atom_map': sorted(map_data['atom_map']),
        'tasks': [sorted([sorted(pair) for pair in layer]) for layer in map_data['tasks']],
    }, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()[:8]

import torch
import concurrent.futures
from lightning import Fabric
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, PrioritizedSampler

from .game import Game
from .mcts import play_game
from .network import Network


def game_to_tensordict(game, td_steps):
    features, bootstrap_features = [], []
    cvals, lvals, pis, bvals = [], [], [], []
    current_qubits = []
    for i in range(len(game.history)):
        obs = game.make_observation(i)
        bootstrap_obs = game.make_observation(min(i + td_steps, len(game.history)))
        target = game.make_target(i, td_steps)
        features.append(obs['features'].float())
        bootstrap_features.append(bootstrap_obs['features'].float())
        cvals.append(target.correctness_value)
        lvals.append(target.latency_value)
        pis.append(target.policy)
        bvals.append(target.bootstrap_discount)
        current_qubits.append(obs.get('current_qubit', -1))
    return TensorDict({
        ('obs', 'features'): torch.stack(features),
        ('obs', 'current_qubit'): torch.tensor(current_qubits, dtype=torch.long),
        ('bootstrap_obs', 'features'): torch.stack(bootstrap_features),
        ('target', 'correctness_values'): torch.tensor(cvals, dtype=torch.float32),
        ('target', 'latency_values'): torch.tensor(lvals, dtype=torch.float32),
        ('target', 'policies'): torch.tensor(pis, dtype=torch.float32),
        ('target', 'bootstrap_discounts'): torch.tensor(bvals, dtype=torch.float32),
    }, batch_size=len(game.history))


def load_dataset_into_buffer(dataset_dir, env_config, td_steps, buffer,
                             filter_class=None, filter_map_id=None):
    from .data import scan_dataset, load_game_batches
    batch_files = scan_dataset(dataset_dir, filter_class=filter_class,
                               filter_map_id=filter_map_id)
    game_dicts = load_game_batches(batch_files)
    loaded = 0
    for gd in game_dicts:
        game = Game.from_dict(gd, env_config)
        game.cache_observation()  # populate observation_cache for make_observation
        td = game_to_tensordict(game, td_steps)
        buffer.extend(td)
        loaded += 1
    return loaded


def _play_single_game(state_dict, config_dict, tasks, initial_positions, network_config_dict, use_fake):
    torch.set_num_threads(1)
    import ml_collections
    config = ml_collections.ConfigDict(config_dict)
    network_config = ml_collections.ConfigDict(network_config_dict)
    net = Network(network_config, use_fake=use_fake)
    if not use_fake:
        net.load_state_dict(state_dict)
    net.eval()
    game = Game(config, tasks, initial_positions)
    game = play_game(game, config.mcts, net)
    return game


class AlphaAtomsTrainer:

    def __init__(self, network, config, tasks, initial_positions):
        self.network = network
        self.config = config
        self.tasks = tasks
        self.initial_positions = initial_positions
        self.board_h = config.env.board_height
        self.board_w = config.env.board_width
        # Map pool: list of (tasks, ip) for curriculum. Index 0 is always the fixed eval map.
        self.map_pool = [(tasks, initial_positions)]
        self.fixed_map_fraction = getattr(config.training, 'fixed_map_fraction', 0.0)
        priority_exponent = getattr(config.training, 'priority_exponent', 0.0)
        if priority_exponent > 0:
            sampler = PrioritizedSampler(
                max_capacity=config.training.buffer_size, alpha=1.0, beta=0.0
            )
        else:
            sampler = None
        self.priority_exponent = priority_exponent
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(config.training.buffer_size),
            **({"sampler": sampler} if sampler else {})
        )
        self.selfplay_iter = 0
        self._fabric = None
        self._model = None
        self._optimizer = None
        self._pool = None
        aug = getattr(config.training, 'data_augmentation', False)
        self._sym_perms = self._make_sym_perms() if aug else None

    def _make_sym_perms(self):
        """(8, board_size) symmetry permutations for a square board. None if non-square."""
        N = self.board_h
        if self.board_h != self.board_w:
            return None
        r = torch.arange(N).repeat_interleave(N)  # row index for each cell
        c = torch.arange(N).repeat(N)              # col index for each cell
        perms = []
        for rot in range(4):
            for flip in [False, True]:
                pr, pc = r.clone(), c.clone()
                if flip:
                    pc = N - 1 - pc
                for _ in range(rot):
                    pr, pc = N - 1 - pc, pr  # undo 90°CW = apply 90°CCW
                perms.append(pr * N + pc)
        return torch.stack(perms)  # (8, board_size)

    def _setup_fabric(self):
        if self._fabric is not None:
            return
        cfg = self.config.training
        self._fabric = Fabric(accelerator=cfg.accelerator, devices=cfg.devices)
        self._fabric.seed_everything(cfg.seed)
        self._fabric.launch()
        self._optimizer = torch.optim.AdamW(self.network.parameters(), lr=cfg.lr)
        self._model, self._optimizer = self._fabric.setup(self.network, self._optimizer)
        if hasattr(self._model, 't_nnet') and hasattr(self._model.t_nnet, 'to'):
            self._model.t_nnet.to(self._fabric.device)

    def _get_pool(self, num_workers):
        if self._pool is None:
            ctx = torch.multiprocessing.get_context('forkserver')
            self._pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers, mp_context=ctx
            )
        return self._pool

    def _get_game_instance(self):
        import random
        # Curriculum pool: fixed_map_fraction goes to ALL old maps (pool[:-1]) uniformly,
        # remaining 1-fixed_map_fraction focuses on the newest map (pool[-1])
        if self.fixed_map_fraction > 0 and len(self.map_pool) > 1:
            if random.random() < self.fixed_map_fraction:
                return random.choice(self.map_pool[:-1])  # retention: any old map
            else:
                return self.map_pool[-1]  # focus: newest map
        if self.config.random_board:
            from .map_generator import generate_random_map
            from .config import atom_map_to_positions
            seed = self.config.random_board_seed
            if seed < 0:
                seed = None  # truly random
            # Fixed structure matching reference map: same board, qubits, layers, gates per layer
            gates_per_layer = max(len(layer) for layer in self.tasks)
            m = generate_random_map(
                (self.board_h, self.board_w),
                self.config.env.num_qubits,
                self.config.network.num_tasks,
                gates_per_layer=gates_per_layer,
                seed=seed,
            )
            tasks = m['tasks']
            ip = atom_map_to_positions(m['atom_map'], self.board_w)
            return tasks, ip
        return self.tasks, self.initial_positions

    def expand_map_pool(self, map_seed):
        from .map_generator import generate_random_map
        from .config import atom_map_to_positions
        gates_per_layer = max(len(layer) for layer in self.tasks)
        m = generate_random_map(
            (self.board_h, self.board_w),
            self.config.env.num_qubits,
            self.config.network.num_tasks,
            gates_per_layer=gates_per_layer,
            seed=map_seed,
        )
        new_entry = (m['tasks'], atom_map_to_positions(m['atom_map'], self.board_w))
        self.map_pool.append(new_entry)
        return new_entry

    def run_selfplay(self):
        self.network.eval()
        num_games = self.config.training.num_selfplay
        num_parallel = self.config.training.num_parallel_games

        if num_parallel > 1:
            games = self._run_parallel_selfplay(num_games, num_parallel)
        else:
            games = self._run_sequential_selfplay(num_games)

        self.selfplay_iter += 1
        return games

    def _run_sequential_selfplay(self, num_games):
        from .experiment import compute_solution_cost
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        games = []
        for idx in range(num_games):
            tasks, ip = self._get_game_instance()
            game = Game(self.config, tasks, ip)
            game = play_game(game, self.config.mcts, self.network)
            game_cost = compute_solution_cost(game)
            self.save_game(game, game_cost)
            games.append(game)
            tasks_done = game.last_info.get('tasks_done', 0)
            print(f"  game {idx+1}/{num_games}: "
                  f"{len(game.history)} steps, "
                  f"tasks_done={tasks_done}/{len(tasks)}")
        torch.set_num_threads(prev_threads)
        return games

    def _run_parallel_selfplay(self, num_games, num_parallel):
        state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        config_dict = self.config.to_dict()
        network_config_dict = self.config.network.to_dict()
        use_fake = self.network.use_fake

        pool = self._get_pool(num_parallel)
        futures = []
        for _ in range(num_games):
            tasks, ip = self._get_game_instance()
            futures.append(
                pool.submit(_play_single_game, state_dict, config_dict, tasks,
                            ip, network_config_dict, use_fake)
            )

        from .experiment import compute_solution_cost
        games = []
        for idx, future in enumerate(futures):
            game = future.result()
            game_cost = compute_solution_cost(game)
            self.save_game(game, game_cost)
            games.append(game)
            tasks_done = game.last_info.get('tasks_done', 0)
            print(f"  game {idx+1}/{num_games}: "
                  f"{len(game.history)} steps, "
                  f"tasks_done={tasks_done}/{len(game.tasks)}")
        return games

    def save_game(self, game, game_cost=None):
        observations = game_to_tensordict(game, self.config.training.td_steps)
        indices = self.replay_buffer.extend(observations)
        if self.priority_exponent > 0 and game_cost is not None and game_cost > 0:
            priority = (1.0 / game_cost) ** self.priority_exponent
            self.replay_buffer.update_priority(
                indices, torch.full((len(indices),), priority)
            )

    def fit(self):
        cfg = self.config.training
        if self.network.use_fake:
            print("  (FakeNet: skipping training)")
            return None
        if len(self.replay_buffer) < cfg.batch_size:
            print(f"  buffer too small ({len(self.replay_buffer)}), skipping training")
            return None

        self._setup_fabric()
        fabric, model, optimizer = self._fabric, self._model, self._optimizer

        model.train()
        last_losses = None
        for iteration in range(cfg.training_steps):
            batch = self.replay_buffer.sample(cfg.batch_size)
            batch = batch.to(fabric.device)
            if self._sym_perms is not None:
                B = batch['obs']['features'].shape[0]
                perms = self._sym_perms[torch.randint(8, (B,))].to(fabric.device)  # (B, board_size)
                feat_idx = perms[:, None, :, None].expand_as(batch['obs']['features'])
                batch['obs']['features'] = torch.gather(batch['obs']['features'], 2, feat_idx)
                boot = batch['bootstrap_obs']['features']
                batch['bootstrap_obs']['features'] = torch.gather(boot, 2, feat_idx.expand_as(boot))
                batch['target']['policies'] = torch.gather(batch['target']['policies'], 1, perms)
            losses = model(batch)
            optimizer.zero_grad()
            fabric.backward(losses['total'])
            grad_norm = sum(
                p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
            ) ** 0.5
            fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_norm_clip)
            optimizer.step()
            model.t_nnet.update(model.nnet.parameters())
            model._training_steps += 1
            last_losses = {k: (v.item() if hasattr(v, 'item') else v) for k, v in losses.items()}
            last_losses['grad_norm'] = round(grad_norm, 4)

            if (iteration + 1) % cfg.log_interval == 0:
                print(f"  step {iteration+1}/{cfg.training_steps}, "
                      f"loss: {last_losses['total']:.4f} "
                      f"(pi={last_losses['policy']:.3f} "
                      f"cv={last_losses['correctness']:.3f} "
                      f"lv={last_losses['latency']:.3f})")

        model.eval()
        return last_losses

    def save_checkpoint(self, path):
        if self.network.use_fake:
            return
        if self._fabric is not None:
            state = {"model": self._model, "optimizer": self._optimizer}
            self._fabric.save(path, state)
        else:
            torch.save({'model': self.network.state_dict()}, path)

import torch
import concurrent.futures
from lightning import Fabric
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from .game import Game
from .mcts import play_game
from .network import Network


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
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(config.training.buffer_size)
        )
        self.selfplay_iter = 0
        self._fabric = None
        self._model = None
        self._optimizer = None
        self._pool = None

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
        if self.config.random_board:
            from .map_generator import generate_random_map
            from .config import atom_map_to_positions
            import random
            seed = self.config.random_board_seed
            if seed < 0:
                seed = None  # truly random
            # Curriculum: vary gates_per_layer for mixed difficulty
            num_qubits = self.config.env.num_qubits
            max_gates = num_qubits // 2
            gates_per_layer = random.randint(max(2, max_gates // 3), max_gates)
            m = generate_random_map(
                (self.board_h, self.board_w),
                num_qubits,
                self.config.network.num_tasks,
                gates_per_layer=gates_per_layer,
                seed=seed,
            )
            tasks = m['tasks']
            ip = atom_map_to_positions(m['atom_map'], self.board_w)
            return tasks, ip
        return self.tasks, self.initial_positions

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
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        games = []
        for idx in range(num_games):
            tasks, ip = self._get_game_instance()
            game = Game(self.config, tasks, ip)
            game = play_game(game, self.config.mcts, self.network)
            self.save_game(game)
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

        games = []
        for idx, future in enumerate(futures):
            game = future.result()
            self.save_game(game)
            games.append(game)
            tasks_done = game.last_info.get('tasks_done', 0)
            print(f"  game {idx+1}/{num_games}: "
                  f"{len(game.history)} steps, "
                  f"tasks_done={tasks_done}/{len(game.tasks)}")
        return games

    def save_game(self, game):
        td_steps = self.config.training.td_steps
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
        feat_tensor = torch.stack(features)
        boot_tensor = torch.stack(bootstrap_features)
        pi_tensor = torch.tensor(pis, dtype=torch.float32)
        cval_tensor = torch.tensor(cvals, dtype=torch.float32)
        lval_tensor = torch.tensor(lvals, dtype=torch.float32)
        bval_tensor = torch.tensor(bvals, dtype=torch.float32)
        qubit_tensor = torch.tensor(current_qubits, dtype=torch.long)
        # Store original + augmented copies (same spatial transform per game)
        from .augmentation import augment_features_and_policy, num_transforms
        n_aug = num_transforms(self.board_h, self.board_w)
        for t in range(n_aug):
            aug_feat, aug_pi = augment_features_and_policy(
                feat_tensor, pi_tensor, self.board_h, self.board_w, transform=t)
            aug_boot, _ = augment_features_and_policy(
                boot_tensor, pi_tensor, self.board_h, self.board_w, transform=t)
            observations = TensorDict({
                ('obs', 'features'): aug_feat,
                ('obs', 'current_qubit'): qubit_tensor,
                ('bootstrap_obs', 'features'): aug_boot,
                ('target', 'correctness_values'): cval_tensor,
                ('target', 'latency_values'): lval_tensor,
                ('target', 'policies'): aug_pi,
                ('target', 'bootstrap_discounts'): bval_tensor,
            }, batch_size=len(game.history))
            self.replay_buffer.extend(observations)

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
            losses = model(batch)
            optimizer.zero_grad()
            fabric.backward(losses['total'])
            fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_norm_clip)
            optimizer.step()
            model.t_nnet.update(model.nnet.parameters())
            model._training_steps += 1
            last_losses = {k: (v.item() if hasattr(v, 'item') else v) for k, v in losses.items()}

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

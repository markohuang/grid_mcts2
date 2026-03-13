import torch
from torch.nn import functional as F
import concurrent.futures
from lightning import Fabric
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from .game import Game
from .mcts import play_game
from .network import Network, make_features
from .board import create_board
from .rewards import compute_total_cost
from .types import EMPTY_CELL


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
            game = Game(self.config, self.tasks, self.initial_positions)
            game = play_game(game, self.config.mcts, self.network)
            self.save_game(game)
            games.append(game)
            tasks_done = game.last_info.get('tasks_done', 0)
            print(f"  game {idx+1}/{num_games}: "
                  f"{len(game.history)} steps, "
                  f"tasks_done={tasks_done}/{len(self.tasks)}")
        torch.set_num_threads(prev_threads)
        return games

    def _run_parallel_selfplay(self, num_games, num_parallel):
        state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        config_dict = self.config.to_dict()
        network_config_dict = self.config.network.to_dict()
        use_fake = self.network.use_fake

        pool = self._get_pool(num_parallel)
        futures = [
            pool.submit(_play_single_game, state_dict, config_dict, self.tasks,
                        self.initial_positions, network_config_dict, use_fake)
            for _ in range(num_games)
        ]

        games = []
        for idx, future in enumerate(futures):
            game = future.result()
            self.save_game(game)
            games.append(game)
            tasks_done = game.last_info.get('tasks_done', 0)
            print(f"  game {idx+1}/{num_games}: "
                  f"{len(game.history)} steps, "
                  f"tasks_done={tasks_done}/{len(self.tasks)}")
        return games

    def save_game(self, game):
        td_steps = self.config.training.td_steps
        features, bootstrap_features = [], []
        cvals, lvals, pis, bvals = [], [], [], []
        for i in range(len(game.history)):
            obs = game.make_observation(i)
            bootstrap_obs = game.make_observation(min(i + td_steps, len(game.history)))
            target = game.make_target(i, td_steps, -1)
            features.append(obs['features'].float())
            bootstrap_features.append(bootstrap_obs['features'].float())
            cvals.append(target.correctness_value)
            lvals.append(target.latency_value)
            pis.append(target.policy)
            bvals.append(target.bootstrap_discount)
        observations = TensorDict({
            ('obs', 'features'): torch.stack(features),
            ('bootstrap_obs', 'features'): torch.stack(bootstrap_features),
            ('target', 'correctness_values'): torch.tensor(cvals, dtype=torch.float32),
            ('target', 'latency_values'): torch.tensor(lvals, dtype=torch.float32),
            ('target', 'policies'): torch.tensor(pis, dtype=torch.float32),
            ('target', 'bootstrap_discounts'): torch.tensor(bvals, dtype=torch.float32),
        }, batch_size=len(game.history))
        self.replay_buffer.extend(observations)

    def fit(self, epoch=0):
        cfg = self.config.training
        if self.network.use_fake:
            print("  (FakeNet: skipping training)")
            return None
        if len(self.replay_buffer) < cfg.batch_size:
            print(f"  buffer too small ({len(self.replay_buffer)}), skipping training")
            return None

        self._setup_fabric()
        fabric, model, optimizer = self._fabric, self._model, self._optimizer

        freeze_value = epoch < cfg.freeze_value_epochs
        if freeze_value:
            for p in model.nnet.value_net.parameters():
                p.requires_grad_(False)
            print(f"  (value head frozen, epoch {epoch+1}/{cfg.freeze_value_epochs})")

        aux_weight = cfg.aux_value_weight
        aux_features, aux_costs = None, None
        if aux_weight > 0:
            aux_features, aux_costs = self.generate_random_boards(cfg.aux_value_samples)
            aux_features = aux_features.to(fabric.device)
            aux_costs = aux_costs.to(fabric.device)

        model.train()
        last_losses = None
        for iteration in range(cfg.training_steps):
            batch = self.replay_buffer.sample(cfg.batch_size)
            batch = batch.to(fabric.device)
            losses = model(batch, policy_entropy_weight=cfg.policy_entropy_weight)
            total = losses['total']
            if aux_weight > 0:
                idx = torch.randint(len(aux_costs), (cfg.batch_size,))
                aux_out = model.nnet(aux_features[idx])
                c_target = model.scalar_to_two_hot(aux_costs[idx])
                l_target = model.scalar_to_two_hot(-aux_costs[idx])
                aux_loss = (F.cross_entropy(aux_out[0], c_target)
                            + F.cross_entropy(aux_out[1], l_target))
                total = total + aux_weight * aux_loss
                losses['aux_value'] = aux_loss.item()
            optimizer.zero_grad()
            fabric.backward(total)
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

        if freeze_value:
            for p in model.nnet.value_net.parameters():
                p.requires_grad_(True)

        model.eval()
        return last_losses

    def generate_random_boards(self, n_samples):
        board_h, board_w = self.config.env.board_height, self.config.env.board_width
        board_size = board_h * board_w
        num_qubits = self.config.env.num_qubits
        features_list, costs = [], []
        for _ in range(n_samples):
            perm = torch.randperm(board_size)[:num_qubits]
            positions = [(p.item() // board_w, p.item() % board_w) for p in perm]
            board, atom_positions = create_board(board_h, board_w, positions)
            flat = board.clone().flatten().long()
            flat[flat == EMPTY_CELL] = num_qubits
            onehot = torch.zeros(board_size, num_qubits + 1)
            onehot.scatter_(1, flat.unsqueeze(1), 1)
            feat = make_features({'board_onehot': onehot, 'tasks_done': 0}, self.tasks)
            cost, _ = compute_total_cost(board, atom_positions, self.tasks, 0, [])
            features_list.append(feat)
            costs.append(float(cost))
        return torch.stack(features_list), torch.tensor(costs, dtype=torch.float32)

    def pretrain_value(self, steps):
        self._setup_fabric()
        fabric, model, optimizer = self._fabric, self._model, self._optimizer
        n_samples = self.config.training.aux_value_samples
        features, costs = self.generate_random_boards(n_samples)
        features, costs = features.to(fabric.device), costs.to(fabric.device)
        model.train()
        batch_size = min(self.config.training.batch_size, n_samples)
        for step in range(steps):
            idx = torch.randint(n_samples, (batch_size,))
            feat_batch, cost_batch = features[idx], costs[idx]
            output = model.nnet(feat_batch)
            correctness_logits, latency_logits, _ = output
            c_target = model.scalar_to_two_hot(cost_batch)
            l_target = model.scalar_to_two_hot(-cost_batch)
            loss = (F.cross_entropy(correctness_logits, c_target)
                    + F.cross_entropy(latency_logits, l_target))
            optimizer.zero_grad()
            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, max_norm=self.config.training.grad_norm_clip)
            optimizer.step()
            model.t_nnet.update(model.nnet.parameters())
            if (step + 1) % 100 == 0:
                print(f"  pretrain step {step+1}/{steps}, loss: {loss.item():.4f}")
        model.eval()

    def save_checkpoint(self, path):
        if self.network.use_fake:
            return
        if self._fabric is not None:
            state = {"model": self._model, "optimizer": self._optimizer}
            self._fabric.save(path, state)
        else:
            torch.save({'model': self.network.state_dict()}, path)

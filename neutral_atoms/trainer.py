import torch
import random

from .game import Game
from .mcts import play_game
from .network import Network
from .config import MCTSConfig, TrainingConfig
from .types import EnvConfig, Tasks


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        return self._collate(batch)

    def _collate(self, batch):
        return {
            'obs': {
                'features': torch.stack([s['features'] for s in batch]),
            },
            'bootstrap_obs': {
                'features': torch.stack([s['bootstrap_features'] for s in batch]),
            },
            'target': {
                'correctness_values': torch.tensor(
                    [s['correctness_value'] for s in batch], dtype=torch.float32
                ),
                'latency_values': torch.tensor(
                    [s['latency_value'] for s in batch], dtype=torch.float32
                ),
                'policies': torch.tensor(
                    [s['policy'] for s in batch], dtype=torch.float32
                ),
                'bootstrap_discounts': torch.tensor(
                    [s['bootstrap_discount'] for s in batch], dtype=torch.float32
                ),
            },
        }

    def __len__(self):
        return len(self.buffer)


class AlphaAtomsTrainer:

    def __init__(
        self,
        network: Network,
        mcts_config: MCTSConfig,
        training_config: TrainingConfig,
        env_config: EnvConfig,
        tasks: Tasks,
        initial_positions: list[tuple[int, int]],
        action_space_size: int,
    ):
        self.network = network
        self.mcts_config = mcts_config
        self.training_config = training_config
        self.env_config = env_config
        self.tasks = tasks
        self.initial_positions = initial_positions
        self.action_space_size = action_space_size
        self.replay_buffer = ReplayBuffer(training_config.buffer_size)
        self.selfplay_iter = 0

    def run_selfplay(self):
        cfg = self.training_config
        self.network.eval()
        for idx in range(cfg.num_selfplay):
            game = Game(
                self.tasks, self.initial_positions, self.env_config,
                self.action_space_size, self.mcts_config.discount,
            )
            game = play_game(game, self.mcts_config, self.network)
            self.save_game(game)
            tasks_done = game.last_info.get('tasks_done', 0)
            print(f"  game {idx+1}/{cfg.num_selfplay}: "
                  f"{len(game.history)} steps, "
                  f"tasks_done={tasks_done}/{len(self.tasks)}")
        self.selfplay_iter += 1

    def save_game(self, game):
        td = self.training_config.td_steps
        for i in range(len(game.history)):
            obs = game.make_observation(i)
            bootstrap_obs = game.make_observation(min(i + td, len(game.history)))
            target = game.make_target(i, td, -1)
            self.replay_buffer.push({
                'features': obs['features'].float(),
                'bootstrap_features': bootstrap_obs['features'].float(),
                'correctness_value': target.correctness_value,
                'latency_value': target.latency_value,
                'policy': target.policy,
                'bootstrap_discount': target.bootstrap_discount,
            })

    def fit(self):
        cfg = self.training_config
        if self.network.use_fake:
            print("  (FakeNet: skipping training)")
            return
        if len(self.replay_buffer) < cfg.batch_size:
            print(f"  buffer too small ({len(self.replay_buffer)}), skipping training")
            return

        self.network.train()
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=cfg.lr)

        for iteration in range(cfg.training_steps):
            batch = self.replay_buffer.sample(cfg.batch_size)
            loss = self.network(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.grad_norm_clip)
            optimizer.step()
            self.network.t_nnet.update(self.network.nnet.parameters())
            self.network._training_steps += 1

            if (iteration + 1) % cfg.log_interval == 0:
                print(f"  step {iteration+1}/{cfg.training_steps}, "
                      f"loss: {loss.item():.4f}")

        self.network.eval()

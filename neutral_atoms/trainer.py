import torch
from lightning import Fabric
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from .game import Game
from .mcts import play_game


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

    def run_selfplay(self):
        self.network.eval()
        games = []
        for idx in range(self.config.training.num_selfplay):
            game = Game(self.config, self.tasks, self.initial_positions)
            game = play_game(game, self.config.mcts, self.network)
            self.save_game(game)
            games.append(game)
            tasks_done = game.last_info.get('tasks_done', 0)
            print(f"  game {idx+1}/{self.config.training.num_selfplay}: "
                  f"{len(game.history)} steps, "
                  f"tasks_done={tasks_done}/{len(self.tasks)}")
        self.selfplay_iter += 1
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

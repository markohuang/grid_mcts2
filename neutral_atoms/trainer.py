import os
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

    def run_selfplay(self):
        cfg = self.config.training
        self.network.eval()
        for idx in range(cfg.num_selfplay):
            game = Game(self.config, self.tasks, self.initial_positions)
            game = play_game(game, self.config.mcts, self.network)
            self.save_game(game)
            tasks_done = game.last_info.get('tasks_done', 0)
            print(f"  game {idx+1}/{cfg.num_selfplay}: "
                  f"{len(game.history)} steps, "
                  f"tasks_done={tasks_done}/{len(self.tasks)}")
        self.selfplay_iter += 1

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

    def fit(self, logger=None):
        cfg = self.config.training
        if self.network.use_fake:
            print("  (FakeNet: skipping training)")
            return
        if len(self.replay_buffer) < cfg.batch_size:
            print(f"  buffer too small ({len(self.replay_buffer)}), skipping training")
            return

        loggers = [logger] if logger is not None else []
        fabric = Fabric(accelerator=cfg.accelerator, devices=cfg.devices, loggers=loggers)
        fabric.seed_everything(cfg.seed)
        fabric.launch()

        optimizer = torch.optim.AdamW(self.network.parameters(), lr=cfg.lr)
        model, optimizer = fabric.setup(self.network, optimizer)

        if hasattr(model, 't_nnet') and hasattr(model.t_nnet, 'to'):
            model.t_nnet.to(fabric.device)

        model.train()
        for iteration in range(cfg.training_steps):
            batch = self.replay_buffer.sample(cfg.batch_size)
            batch = batch.to(fabric.device)
            loss = model(batch)
            optimizer.zero_grad()
            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_norm_clip)
            optimizer.step()
            model.t_nnet.update(model.nnet.parameters())
            model._training_steps += 1

            if (iteration + 1) % cfg.log_interval == 0:
                if loggers:
                    fabric.log_dict({"train/loss": loss.item()})
                print(f"  step {iteration+1}/{cfg.training_steps}, "
                      f"loss: {loss.item():.4f}")

        model.eval()
        os.makedirs(cfg.save_dir, exist_ok=True)
        state = {"model": model, "optimizer": optimizer}
        fabric.save(os.path.join(cfg.save_dir, f"checkpoint_{self.selfplay_iter}.ckpt"), state)

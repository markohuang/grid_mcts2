"""Pretrain the value network to predict remaining cost from random states.
Then run MCTS training with Option C (layer-completion reward).

Usage:
    ../grid_mcts2/.venv/bin/python pretrain_value.py --config.map_num=2 \
        --config.env.reward_mode=layer_completion --config.training.epochs=50
"""
import os
import time
import random
import torch
from torch.nn import functional as F
from absl import app
from ml_collections import config_flags

from neutral_atoms.config import get_config, atom_map_to_positions, set_derived_config, get_map_data
from neutral_atoms.map_generator import generate_random_map
from neutral_atoms.env import NeutralAtomsEnv
from neutral_atoms.network import Network, make_features
from neutral_atoms.rewards import compute_total_cost

_CONFIG = config_flags.DEFINE_config_dict('config', get_config())

PRETRAIN_STEPS = 2000
PRETRAIN_BATCH = 64
PRETRAIN_LR = 1e-3


def generate_pretrain_batch(config, batch_size):
    features_list, cost_targets = [], []
    board_h, board_w = config.env.board_height, config.env.board_width
    num_qubits = config.env.num_qubits
    num_tasks = config.network.num_tasks

    for _ in range(batch_size):
        m = generate_random_map((board_h, board_w), num_qubits, num_tasks,
                                gates_per_layer=random.randint(2, num_qubits // 2))
        ip = atom_map_to_positions(m['atom_map'], board_w)
        env = NeutralAtomsEnv(m['tasks'], ip, config.env)
        env.reset()
        # Random partial progress
        n_steps = random.randint(0, env.episode_length - 1)
        for _ in range(n_steps):
            if env.tasks_done >= env.num_tasks:
                break
            env.step(random.choice(env.legal_actions()))
        obs = env._get_observation()
        feat = make_features(obs, m['tasks'])
        remaining = compute_total_cost(env.board, env.atom_positions, m['tasks'],
                                       env.tasks_done, env.current_phase_moves)
        features_list.append(feat.float())
        cost_targets.append(-float(remaining))

    return torch.stack(features_list), torch.tensor(cost_targets, dtype=torch.float32)


def pretrain(network, config):
    print(f"=== Pretraining value network ({PRETRAIN_STEPS} steps) ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = network.to(device)
    network.train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=PRETRAIN_LR)

    for step in range(PRETRAIN_STEPS):
        features, targets = generate_pretrain_batch(config, PRETRAIN_BATCH)
        features, targets = features.to(device), targets.to(device)
        targets = targets.clip(network.cfg.value_min, network.cfg.value_max)

        # Forward through value network only
        pred = network.inference({'features': features})
        pred_cv = network.logits2values(pred.correctness_value_logits)
        loss = F.mse_loss(pred_cv, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(f"  step {step+1}/{PRETRAIN_STEPS}: loss={loss.item():.3f}, "
                  f"pred_mean={pred_cv.mean().item():.2f}, target_mean={targets.mean().item():.2f}")

    network = network.cpu()
    network.eval()
    print(f"Pretraining complete. Final loss: {loss.item():.3f}")
    return network


def main(_):
    from neutral_atoms.trainer import AlphaAtomsTrainer
    from neutral_atoms.experiment import (
        create_run_dir, selfplay_metrics, compute_solution_cost, save_solution,
        log_game_trace, append_epoch_metrics, append_to_registry,
    )

    config = _CONFIG.value
    set_derived_config(config)
    map_data = get_map_data(config)
    tasks = map_data['tasks']
    initial_positions = atom_map_to_positions(map_data['atom_map'], config.env.board_width)

    network = Network(config.network, use_fake=config.use_fake)
    if not config.use_fake:
        network = pretrain(network, config)

    # Standard training loop from main.py
    from main import print_config_summary, format_selfplay_summary
    print_config_summary(config, network)

    run_id, run_dir = create_run_dir(config)
    print(f"Run: {run_id} -> {run_dir}")

    trainer = AlphaAtomsTrainer(network, config, tasks, initial_positions)

    best_cost, best_game = float('inf'), None
    eval_best_cost = float('inf')
    no_improve_count = 0
    run_metrics = {}

    eval_tasks = tasks
    eval_positions = initial_positions

    for epoch in range(config.training.epochs):
        print(f"\n=== Epoch {epoch+1}/{config.training.epochs} ===")

        print("Self-play:")
        t0 = time.time()
        games = trainer.run_selfplay()
        selfplay_time = time.time() - t0
        sp_metrics = selfplay_metrics(games)
        print(f"  >> {format_selfplay_summary(sp_metrics)} [{selfplay_time:.1f}s]")

        if 'best_cost' in sp_metrics:
            if sp_metrics['best_cost'] < best_cost:
                best_cost = sp_metrics['best_cost']

        from neutral_atoms.game import Game
        from neutral_atoms.mcts import play_game
        trainer.network.eval()
        eval_games, eval_costs = [], []
        for _ in range(5):
            eg = Game(config, eval_tasks, eval_positions)
            eg = play_game(eg, config.mcts, trainer.network)
            eval_games.append(eg)
            eval_costs.append(compute_solution_cost(eg))
        eval_cost = min(eval_costs)
        eval_avg = sum(eval_costs) / len(eval_costs)
        if eval_cost < eval_best_cost:
            eval_best_cost = eval_cost
            best_game = eval_games[eval_costs.index(eval_cost)]
            no_improve_count = 0
        else:
            no_improve_count += 1
        print(f"  eval: best={eval_cost}, avg={eval_avg:.1f}, best_so_far={eval_best_cost}")

        print("Training:")
        t0 = time.time()
        train_result = trainer.fit()
        train_time = time.time() - t0

        epoch_metrics = {
            'epoch': epoch + 1,
            'selfplay_time': round(selfplay_time, 2),
            'train_time': round(train_time, 2),
            **{k: v for k, v in sp_metrics.items() if k != 'best_game'},
            'eval_best': eval_cost,
            'eval_avg': round(eval_avg, 1),
            'eval_best_so_far': eval_best_cost,
        }
        if train_result:
            epoch_metrics.update({f'train_{k}': v for k, v in train_result.items()})
        append_epoch_metrics(run_dir, epoch_metrics)

        if (epoch + 1) % config.experiment.checkpoint_every_n_epochs == 0:
            ckpt_path = os.path.join(run_dir, 'checkpoints', f'epoch_{epoch+1:03d}.ckpt')
            trainer.save_checkpoint(ckpt_path, run_id=run_id, epoch=epoch + 1)

        if no_improve_count >= config.experiment.early_stopping_patience:
            print(f"\nEarly stopping: no improvement for {no_improve_count} epochs")
            break

    trainer.save_checkpoint(os.path.join(run_dir, 'checkpoints', 'final.ckpt'),
                            run_id=run_id, epoch=epoch + 1)
    if best_game is not None:
        save_solution(best_game, os.path.join(run_dir, 'solutions', 'best.json'))
        log_game_trace(best_game, run_dir, label='best')
        print(f"\nBest eval cost: {eval_best_cost}")


if __name__ == '__main__':
    app.run(main)

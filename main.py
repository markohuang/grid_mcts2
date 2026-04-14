import os
import time
from absl import app
from ml_collections import config_flags

from neutral_atoms.config import get_config, MAPS, atom_map_to_positions, set_derived_config, get_map_data
from neutral_atoms.network import Network
from neutral_atoms.trainer import AlphaAtomsTrainer
from neutral_atoms.experiment import (
    create_run_dir, selfplay_metrics, compute_solution_cost, save_solution,
    log_game_trace, append_epoch_metrics, append_to_registry,
)

_CONFIG = config_flags.DEFINE_config_dict('config', get_config())


def print_config_summary(config, network):
    print(f"Map {config.map_num}: {config.env.board_height}x{config.env.board_width}, "
          f"{config.env.num_qubits} qubits, {config.network.num_tasks} tasks")
    print(f"Action space: {config.network.num_actions} (board_size), Use fake: {config.use_fake}")
    if not config.use_fake:
        n_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        print(f"Trainable params: {n_params}, "
              f"Sims: {config.mcts.num_simulations}, "
              f"Parallel: {config.training.num_parallel_games}")


def format_selfplay_summary(sp_metrics):
    parts = [f"best={sp_metrics['best_cost']}", f"avg={sp_metrics['avg_cost']}±{sp_metrics['cost_std']}"]
    parts.append(f"noop={sp_metrics['noop_frac']:.0%}")
    if 'avg_mcts_depth' in sp_metrics:
        parts.append(f"depth={sp_metrics['avg_mcts_depth']}")
    if 'mcts_reward_frac' in sp_metrics:
        parts.append(f"r_frac={sp_metrics['mcts_reward_frac']:.0%}")
    if 'mcts_boundary_reach_frac' in sp_metrics:
        parts.append(f"b_frac={sp_metrics['mcts_boundary_reach_frac']:.0%}")
    if 'mcts_reward_sum_std' in sp_metrics:
        parts.append(f"r_std={sp_metrics['mcts_reward_sum_std']:.2f}")
    if 'avg_root_value' in sp_metrics:
        parts.append(f"V={sp_metrics['avg_root_value']:.2f}")
    if 'avg_policy_entropy' in sp_metrics:
        parts.append(f"H={sp_metrics['avg_policy_entropy']:.2f}")
    return ', '.join(parts)


def main(_):
    config = _CONFIG.value
    set_derived_config(config)
    if config.mcts.prior_mix_weight > 0:
        assert config.env.reward_mode == 'layer_delta', \
            "prior_mix_weight > 0 is only valid with reward_mode='layer_delta' " \
            "(plan_cost mode already puts plan_cost_delta into the reward)."
        with config.env.unlocked():
            config.env.track_plan_delta = True
    map_data = get_map_data(config)
    tasks = map_data['tasks']
    initial_positions = atom_map_to_positions(map_data['atom_map'], config.env.board_width)

    network = Network(config.network, use_fake=config.use_fake)
    if config.experiment.load_checkpoint and not config.use_fake:
        import torch
        ckpt = torch.load(config.experiment.load_checkpoint, map_location='cpu')
        network.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint: {config.experiment.load_checkpoint}")
    print_config_summary(config, network)

    run_id, run_dir = create_run_dir(config)
    print(f"Run: {run_id} -> {run_dir}")

    trainer = AlphaAtomsTrainer(network, config, tasks, initial_positions)

    best_cost, best_game = float('inf'), None
    no_improve_count = 0
    run_metrics = {}
    epochs_completed = 0
    cumulative_time = 0.0
    eval_window = getattr(config.experiment, 'eval_window', 10)
    curriculum_phase = 0
    curriculum_maps_remaining = config.experiment.curriculum_maps
    curriculum_patience = config.experiment.curriculum_patience

    # Per-map eval state: index 0 = reference map, grows with curriculum
    eval_map_pool = [(tasks, initial_positions)]  # (tasks, ip) pairs
    eval_best_per_map = [float('inf')]
    eval_best_epoch_per_map = [None]
    eval_avg_history_per_map = [[]]

    from neutral_atoms.game import Game
    from neutral_atoms.mcts import play_game

    # Pre-populate map pool when resuming mid-curriculum (uses same deterministic seeds)
    for phase in range(1, config.experiment.curriculum_initial_phase + 1):
        map_seed = config.training.seed * 100 + phase
        new_entry = trainer.expand_map_pool(map_seed)
        eval_map_pool.append(new_entry)
        eval_best_per_map.append(float('inf'))
        eval_best_epoch_per_map.append(None)
        eval_avg_history_per_map.append([])
        curriculum_phase = phase
        curriculum_maps_remaining -= 1
    if config.experiment.curriculum_initial_phase > 0:
        print(f"Resumed at curriculum phase {curriculum_phase}, pool_size={len(trainer.map_pool)}")

    for epoch in range(config.training.epochs):
        epochs_completed = epoch + 1
        print(f"\n=== Epoch {epochs_completed}/{config.training.epochs} ===")

        print("Self-play:")
        t0 = time.time()
        games = trainer.run_selfplay()
        selfplay_time = time.time() - t0
        sp_metrics = selfplay_metrics(games)
        print(f"  >> {format_selfplay_summary(sp_metrics)} [{selfplay_time:.1f}s]")

        if 'best_cost' in sp_metrics and sp_metrics['best_cost'] < best_cost:
            best_cost = sp_metrics['best_cost']
            best_game = sp_metrics['best_game']

        # Evaluate each map in eval_map_pool (5 games each, no noise)
        trainer.network.eval()
        eval_parts = []
        per_map_metrics = {}
        for map_idx, (ev_tasks, ev_positions) in enumerate(eval_map_pool):
            ev_games, ev_costs = [], []
            for _ in range(5):
                eg = Game(config, ev_tasks, ev_positions)
                eg = play_game(eg, config.mcts, trainer.network,
                               add_exploration_noise=False, deterministic=True)
                ev_games.append(eg)
                ev_costs.append(compute_solution_cost(eg))
            ev_best = min(ev_costs)
            ev_avg = sum(ev_costs) / len(ev_costs)
            eval_avg_history_per_map[map_idx].append(ev_avg)
            ev_ravg = (sum(eval_avg_history_per_map[map_idx][-eval_window:])
                       / len(eval_avg_history_per_map[map_idx][-eval_window:]))
            improved = ev_best < eval_best_per_map[map_idx]
            if improved:
                eval_best_per_map[map_idx] = ev_best
                eval_best_epoch_per_map[map_idx] = epochs_completed
                if map_idx == 0:
                    best_game = ev_games[ev_costs.index(ev_best)]
            eval_parts.append(
                f"map{map_idx}: best={ev_best}, avg={ev_avg:.1f}, "
                f"ravg={ev_ravg:.1f}, bf={eval_best_per_map[map_idx]} (E{eval_best_epoch_per_map[map_idx]})"
            )
            per_map_metrics[f'eval_map{map_idx}_best'] = ev_best
            per_map_metrics[f'eval_map{map_idx}_avg'] = round(ev_avg, 1)
            per_map_metrics[f'eval_map{map_idx}_ravg'] = round(ev_ravg, 2)
            per_map_metrics[f'eval_map{map_idx}_best_so_far'] = eval_best_per_map[map_idx]
        print(f"  eval: {' | '.join(eval_parts)}")

        # Early stopping / curriculum trigger: track improvement on newest map
        if eval_best_epoch_per_map[-1] == epochs_completed:
            no_improve_count = 0
        else:
            no_improve_count += 1

        print("Training:")
        t0 = time.time()
        train_result = trainer.fit()
        train_time = time.time() - t0

        epoch_time = round(selfplay_time + train_time, 2)
        cumulative_time += epoch_time

        epoch_metrics = {
            'epoch': epochs_completed,
            'selfplay_time': round(selfplay_time, 2),
            'train_time': round(train_time, 2),
            'epoch_time': epoch_time,
            'cumulative_time': round(cumulative_time, 2),
            **{k: v for k, v in sp_metrics.items() if k != 'best_game'},
            'best_cost_so_far': best_cost if best_cost < float('inf') else None,
            'curriculum_phase': curriculum_phase,
            'pool_size': len(trainer.map_pool),
            **per_map_metrics,
        }
        if train_result:
            epoch_metrics.update({f'train_{k}': v for k, v in train_result.items()})
            run_metrics['final_loss'] = train_result['total']
        append_epoch_metrics(run_dir, epoch_metrics)

        if (epoch + 1) % config.experiment.checkpoint_every_n_epochs == 0:
            ckpt_path = os.path.join(run_dir, 'checkpoints', f'epoch_{epoch+1:03d}.ckpt')
            trainer.save_checkpoint(ckpt_path, run_id=run_id, epoch=epoch + 1)
            print(f"  checkpoint: epoch_{epoch+1:03d}.ckpt")

        # Curriculum expansion: when stalled on newest map, add a new map
        if curriculum_maps_remaining > 0 and no_improve_count >= curriculum_patience:
            curriculum_phase += 1
            map_seed = config.training.seed * 100 + curriculum_phase
            new_entry = trainer.expand_map_pool(map_seed)
            eval_map_pool.append(new_entry)
            eval_best_per_map.append(float('inf'))
            eval_best_epoch_per_map.append(None)
            eval_avg_history_per_map.append([])
            curriculum_maps_remaining -= 1
            no_improve_count = 0
            print(f"\nCurriculum: phase {curriculum_phase}, pool_size={len(trainer.map_pool)} "
                  f"(seed={map_seed}), {curriculum_maps_remaining} expansions left")
        elif no_improve_count >= config.experiment.early_stopping_patience:
            print(f"\nEarly stopping: no improvement for {no_improve_count} epochs")
            break

    # Save final artifacts
    trainer.save_checkpoint(os.path.join(run_dir, 'checkpoints', 'final.ckpt'),
                            run_id=run_id, epoch=epochs_completed)

    eval_best_cost = eval_best_per_map[0]  # reference map best cost for summary
    eval_best_epoch = eval_best_epoch_per_map[0]
    if best_game is not None:
        save_solution(best_game, os.path.join(run_dir, 'solutions', 'best.json'))
        log_game_trace(best_game, run_dir, label='best')
        print(f"\nBest eval cost (map0): {eval_best_cost} (epoch {eval_best_epoch})")

    run_metrics.update({
        'best_cost': eval_best_cost if eval_best_cost < float('inf') else None,
        'best_cost_epoch': eval_best_epoch,
        'avg_cost': sp_metrics.get('avg_cost'),
        'completion_rate': sp_metrics.get('completion_rate', 0),
        'epochs_completed': epochs_completed,
        'total_time': round(cumulative_time, 2),
        **{f'eval_map{i}_best_final': eval_best_per_map[i] for i in range(len(eval_map_pool))},
    })
    append_to_registry(config.experiment.output_dir, run_id, config, run_metrics)
    print(f"Run {run_id} complete")


if __name__ == '__main__':
    app.run(main)

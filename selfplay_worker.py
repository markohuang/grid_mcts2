"""Distributed self-play worker for HPC (Narval).

Generates MCTS games across multiple CPU cores and saves compact game dicts
to a shared dataset directory. Designed to run as a SLURM job array task.

Usage:
  python selfplay_worker.py \
    --config.map_num=2 \
    --config.mcts.num_simulations=1000 \
    --dataset_dir=/project/.../datasets/run01 \
    --num_games=120 \
    --num_workers=60 \
    --batch_size=60 \
    --weights_path=weights/latest.pt  # optional, omit for random policy
"""

import json
import math
import os
import sys
import time
import socket
import argparse
import torch
import concurrent.futures

from neutral_atoms.config import (
    set_derived_config, get_map_data,
    atom_map_to_positions, map_class, map_id, MAPS,
)
from neutral_atoms.network import Network
from neutral_atoms.game import Game
from fast_mcts.backends import get_backend
from neutral_atoms.data import save_game_batch, save_map_spec, save_manifest, atomic_save, append_index
from neutral_atoms.experiment import compute_solution_cost, _git_metadata


# lineage fields we copy from a loaded checkpoint into each game dict
_CKPT_LINEAGE_FIELDS = ('run_id', 'parent_run', 'parent_ckpt', 'epoch', 'training_steps',
                        'git_sha', 'git_dirty', 'config_hash')


def _play_single_game(state_dict, config_dict, tasks, initial_positions,
                      network_config_dict, use_fake, training_steps,
                      reference_map=None):
    torch.set_num_threads(1)
    import ml_collections
    config = ml_collections.ConfigDict(config_dict)
    network_config = ml_collections.ConfigDict(network_config_dict)
    net = Network(network_config, use_fake=use_fake)
    if not use_fake and state_dict is not None:
        net.load_state_dict(state_dict)
    net.eval()
    # If random_board is on, regenerate a fresh map for THIS game only. Each worker
    # draws independently; truly random per call when random_board_seed=-1.
    if config.random_board:
        from neutral_atoms.map_generator import generate_random_map
        from neutral_atoms.config import atom_map_to_positions
        ref = reference_map
        gates_per_layer = max(len(layer) for layer in ref['tasks'])
        seed = config.random_board_seed if config.random_board_seed >= 0 else None
        m = generate_random_map(
            ref['board_dim'], ref['num_qubits'], len(ref['tasks']),
            gates_per_layer=gates_per_layer, seed=seed,
        )
        tasks = m['tasks']
        initial_positions = atom_map_to_positions(m['atom_map'], config.env.board_width)
    t0 = time.time()
    game = Game(config, tasks, initial_positions)
    game = get_backend(config.mcts.backend)(game, config.mcts, net)
    game_time = time.time() - t0
    return game, game_time


def _game_metrics(game, game_time):
    noop_count = 0
    board_width = game.env_config.board_width
    for i, action in enumerate(game.history):
        # Replay to find current positions — but we can approximate from last_info
        pass
    noop_frac = game.last_info.get('num_moves', 0)
    total_steps = len(game.history)
    num_moves = game.last_info.get('num_moves', 0)
    avg_policy_entropy = 0.0
    if game.child_visits:
        entropies = []
        for visits in game.child_visits:
            probs = [p for p in visits if p > 0]
            if probs:
                entropies.append(-sum(p * math.log(p) for p in probs))
        if entropies:
            avg_policy_entropy = sum(entropies) / len(entropies)
    avg_root_value = sum(game.root_values) / len(game.root_values) if game.root_values else 0.0
    avg_mcts_depth = sum(game.mcts_depths) / len(game.mcts_depths) if game.mcts_depths else 0.0
    _mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    avg_mcts_reward_std = _mean(game.mcts_reward_sum_stds)
    avg_mcts_reward_sum_mean = _mean(game.mcts_reward_sum_means)
    avg_mcts_reward_abs = _mean(game.mcts_reward_abs_sum_means)
    avg_mcts_boundary_frac = _mean(game.mcts_boundary_reach_fracs)
    avg_mcts_reached_terminal_frac = _mean(game.mcts_reached_terminal_fracs)
    avg_mcts_sign_changes = _mean(game.mcts_sign_changes_means)
    avg_mcts_reward_frac = _mean(game.mcts_reward_fracs)
    return {
        'steps': total_steps,
        'num_moves': num_moves,
        'noop_frac': round(1 - num_moves / total_steps, 3) if total_steps > 0 else 0,
        'move_distance': game.last_info.get('total_move_distance', 0),
        'avg_policy_entropy': round(avg_policy_entropy, 3),
        'avg_root_value': round(avg_root_value, 3),
        'avg_mcts_depth': round(avg_mcts_depth, 1),
        'avg_mcts_reward_std': round(avg_mcts_reward_std, 4),
        'avg_mcts_reward_sum_mean': round(avg_mcts_reward_sum_mean, 4),
        'avg_mcts_reward_abs': round(avg_mcts_reward_abs, 4),
        'avg_mcts_boundary_frac': round(avg_mcts_boundary_frac, 3),
        'avg_mcts_reached_terminal_frac': round(avg_mcts_reached_terminal_frac, 3),
        'avg_mcts_sign_changes': round(avg_mcts_sign_changes, 3),
        'avg_mcts_reward_frac': round(avg_mcts_reward_frac, 3),
        'game_time_s': round(game_time, 2),
        'tasks_done': game.last_info.get('tasks_done', 0),
    }


def _enrich_game_dict(gd, game, game_time, map_data, config, weights_path,
                      training_steps, weight_lineage, selfplay_context):
    # When random_board=True, each game has its own unique map. Derive map_id from
    # the game's actual tasks + initial_positions (not from the shared reference map).
    if config.random_board:
        w = config.env.board_width
        per_game_map = {
            'board_dim': map_data['board_dim'],
            'num_qubits': map_data['num_qubits'],
            'atom_map': [r * w + c for (r, c) in game.initial_positions],
            'tasks': game.tasks,
        }
        gd['map_id'] = map_id(per_game_map)
        gd['map_class'] = map_class(per_game_map)
    else:
        gd['map_id'] = map_id(map_data)
        gd['map_class'] = map_class(map_data)
    gd['cost'] = compute_solution_cost(game)
    gd['num_simulations'] = config.mcts.num_simulations
    gd['weight_gen'] = training_steps
    gd['weight_file'] = os.path.basename(weights_path) if weights_path else 'init'
    gd['weight_lineage'] = weight_lineage
    gd['selfplay_context'] = selfplay_context
    gd['reward_mode'] = config.env.reward_mode
    gd['prior_mix_weight'] = config.mcts.prior_mix_weight
    gd['metrics'] = _game_metrics(game, game_time)
    return gd


def _save_worker_log(dataset_dir, node_id, log_entry):
    log_dir = os.path.join(dataset_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{node_id}.jsonl')
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def run_worker(config, map_data, dataset_dir, num_games, num_workers,
               batch_size, weights_path, node_id, slurm_job_id, slurm_array_task_id):
    tasks = map_data['tasks']
    initial_positions = atom_map_to_positions(map_data['atom_map'],
                                              config.env.board_width)
    # Skip map_spec dump when random_board=True — the reference map is only a template
    # for structure, not the actual map any game uses. Per-game map_id lives in the index.
    if not config.random_board:
        save_map_spec(dataset_dir, map_data)

    network = Network(config.network, use_fake=config.use_fake)
    state_dict = None
    training_steps = 0
    weight_lineage = {k: '' for k in _CKPT_LINEAGE_FIELDS}
    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        if 'training_steps' not in ckpt:
            raise KeyError(
                f"checkpoint {weights_path} has no 'training_steps' field — "
                "re-save with updated trainer (commit after weight_gen fix) or "
                "the temperature schedule will silently snap to step 0"
            )
        training_steps = int(ckpt['training_steps'])
        weight_lineage = {k: ckpt.get(k, '') for k in _CKPT_LINEAGE_FIELDS}
        weight_lineage['weights_path'] = weights_path
        if 'model' in ckpt:
            network.load_state_dict(ckpt['model'])
            state_dict = {k: v.cpu() for k, v in ckpt['model'].items()}
        else:
            network.load_state_dict(ckpt)
            state_dict = {k: v.cpu() for k, v in ckpt.items()}
        print(f"Loaded weights from {weights_path} (training_steps={training_steps}, "
              f"run_id={weight_lineage.get('run_id','')}, git_sha={weight_lineage.get('git_sha','')[:8]})")
    elif not config.use_fake:
        state_dict = {k: v.cpu() for k, v in network.state_dict().items()}
        weight_lineage = {k: '' for k in _CKPT_LINEAGE_FIELDS}
        weight_lineage['weights_path'] = ''
        print("Using randomly initialized network (training_steps=0)")
    else:
        weight_lineage = {k: '' for k in _CKPT_LINEAGE_FIELDS}
        weight_lineage['weights_path'] = ''

    git = _git_metadata()
    selfplay_context = {
        'git_sha': git['git_commit'],
        'git_dirty': git['git_dirty'],
        'slurm_job_id': slurm_job_id,
        'slurm_array_task_id': slurm_array_task_id,
        'node_id': node_id,
        'hostname': socket.gethostname(),
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    config_dict = config.to_dict()
    network_config_dict = config.network.to_dict()

    ctx = torch.multiprocessing.get_context('forkserver')
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, mp_context=ctx
    )

    total_played = 0
    batch_seq = 0
    all_costs = []
    all_game_times = []
    t_start = time.time()

    # For random_board, we use a placeholder map_id at the directory level since each
    # game has a distinct per-game map_id stored in its own dict + the parquet index.
    save_map_id_override = 'mixed' if config.random_board else None

    while total_played < num_games:
        chunk = min(batch_size, num_games - total_played)
        t_batch = time.time()
        futures = []
        for _ in range(chunk):
            futures.append(pool.submit(
                _play_single_game, state_dict, config_dict, tasks,
                initial_positions, network_config_dict, config.use_fake,
                training_steps, map_data if config.random_board else None,
            ))

        game_dicts = []
        batch_costs = []
        for future in futures:
            game, game_time = future.result()
            gd = _enrich_game_dict(
                game.to_dict(), game, game_time, map_data, config, weights_path,
                training_steps, weight_lineage, selfplay_context,
            )
            game_dicts.append(gd)
            batch_costs.append(gd['cost'])
            all_costs.append(gd['cost'])
            all_game_times.append(game_time)
            total_played += 1

        path = save_game_batch(dataset_dir, game_dicts, map_data, node_id, batch_seq,
                               map_id_override=save_map_id_override)
        batch_file_rel = os.path.relpath(path, dataset_dir)
        append_index(dataset_dir, game_dicts, batch_file_rel, node_id, batch_seq)
        batch_elapsed = time.time() - t_batch
        total_elapsed = time.time() - t_start
        rate = total_played / total_elapsed if total_elapsed > 0 else 0
        avg_cost = sum(batch_costs) / len(batch_costs)
        avg_game_time = sum(gt for gd in game_dicts for gt in [gd['metrics']['game_time_s']]) / len(game_dicts)

        # Structured batch log
        batch_log = {
            'batch_seq': batch_seq,
            'num_games': len(game_dicts),
            'batch_time_s': round(batch_elapsed, 2),
            'avg_game_time_s': round(avg_game_time, 2),
            'avg_cost': round(avg_cost, 1),
            'min_cost': min(batch_costs),
            'max_cost': max(batch_costs),
            'total_played': total_played,
            'rate_games_per_s': round(rate, 2),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        _save_worker_log(dataset_dir, node_id, batch_log)

        print(f"  batch {batch_seq}: {len(game_dicts)} games, "
              f"cost={avg_cost:.1f} [{min(batch_costs)}-{max(batch_costs)}], "
              f"game_t={avg_game_time:.1f}s, "
              f"total={total_played}/{num_games}, "
              f"rate={rate:.1f} g/s")
        batch_seq += 1

    pool.shutdown()
    total_elapsed = time.time() - t_start

    # Final summary log
    summary = {
        'node_id': node_id,
        'total_games': total_played,
        'total_batches': batch_seq,
        'total_time_s': round(total_elapsed, 1),
        'games_per_s': round(total_played / total_elapsed, 2) if total_elapsed > 0 else 0,
        'avg_cost': round(sum(all_costs) / len(all_costs), 1) if all_costs else None,
        'min_cost': min(all_costs) if all_costs else None,
        'avg_game_time_s': round(sum(all_game_times) / len(all_game_times), 2) if all_game_times else None,
        'num_simulations': config.mcts.num_simulations,
        'num_workers': num_workers,
        'map_class': map_class(map_data),
        'map_id': map_id(map_data),
        'use_fake': config.use_fake,
        'hostname': socket.gethostname(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    summary_path = os.path.join(dataset_dir, 'logs', f'{node_id}_summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Worker Summary ===")
    print(f"  {total_played} games in {total_elapsed:.1f}s "
          f"({total_played/total_elapsed:.1f} games/s)")
    print(f"  cost: avg={summary['avg_cost']}, min={summary['min_cost']}")
    print(f"  avg game time: {summary['avg_game_time_s']}s")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--num_games', type=int, default=120)
    parser.add_argument('--num_workers', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--weights_path', default='')
    parser.add_argument('--node_id', default='')
    parser.add_argument('--notes', default='')
    parser.add_argument('--preset', choices=['default', 'hpc'], default='default',
                        help="default=config.py (smoke-test knobs); "
                             "hpc=config_hpc.py (800 sims, scale-up defaults)")
    parser.add_argument('--slurm_job_id', default=os.environ.get('SLURM_JOB_ID', ''))
    parser.add_argument('--slurm_array_task_id',
                        default=os.environ.get('SLURM_ARRAY_TASK_ID', ''))
    args, remaining = parser.parse_known_args()

    if args.preset == 'hpc':
        from neutral_atoms.config_hpc import get_config as _get_config
    else:
        from neutral_atoms.config import get_config as _get_config

    sys.argv = [sys.argv[0]] + remaining
    from absl import app
    from ml_collections import config_flags
    _CONFIG = config_flags.DEFINE_config_dict('config', _get_config())

    def _main(_):
        config = _CONFIG.value
        set_derived_config(config)
        map_data = get_map_data(config)

        node_id = args.node_id or f"{socket.gethostname()}_{os.getpid()}"
        print(f"Self-play worker: node_id={node_id} preset={args.preset}")
        print(f"  map_class={map_class(map_data)}, map_id={map_id(map_data)}")
        print(f"  num_games={args.num_games}, num_workers={args.num_workers}, "
              f"sims={config.mcts.num_simulations}, use_fake={config.use_fake}")
        print(f"  dataset_dir={args.dataset_dir}")
        print(f"  slurm: job_id={args.slurm_job_id}, array_task={args.slurm_array_task_id}")

        save_manifest(args.dataset_dir, config.to_dict(), notes=args.notes)

        run_worker(
            config, map_data, args.dataset_dir,
            num_games=args.num_games,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            weights_path=args.weights_path,
            node_id=node_id,
            slurm_job_id=args.slurm_job_id,
            slurm_array_task_id=args.slurm_array_task_id,
        )

    app.run(_main)


if __name__ == '__main__':
    main()

import json
import math
import os
import hashlib
import time
import sys
import subprocess
import torch

from .env import NeutralAtomsEnv
from .tasks import gates_to_moves, get_relevant_atoms
from .moves import count_groups


def _split_tags(raw_tags):
    return [tag.strip() for tag in raw_tags.split(',') if tag.strip()]


def _git_metadata():
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return {'git_commit': '', 'git_dirty': None}
    try:
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], text=True, stderr=subprocess.DEVNULL
        ).strip())
    except Exception:
        dirty = None
    return {'git_commit': commit, 'git_dirty': dirty}


def build_run_manifest(run_id, config):
    manifest = {
        'run_id': run_id,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'argv': sys.argv,
        'cwd': os.getcwd(),
        'study': config.experiment.study,
        'hypothesis': config.experiment.hypothesis,
        'variant': config.experiment.variant,
        'tags': _split_tags(config.experiment.tags),
        'notes': config.experiment.notes,
        'decision': config.experiment.decision,
        'parent_run': config.experiment.parent_run,
        'map_num': config.map_num,
        'reward_mode': config.env.reward_mode,
        'random_board': config.random_board,
        'use_fake': config.use_fake,
        'num_simulations': config.mcts.num_simulations,
        'plan_cost_search_bonus_weight': config.mcts.plan_cost_search_bonus_weight,
        'seed': config.training.seed,
    }
    manifest.update(_git_metadata())
    return manifest


def create_run_dir(config):
    run_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
    run_dir = os.path.join(config.experiment.output_dir, run_id)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'solutions'), exist_ok=True)
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    manifest_path = os.path.join(run_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(build_run_manifest(run_id, config), f, indent=2)
    return run_id, run_dir


def compute_solution_cost(game):
    env = NeutralAtomsEnv(game.tasks, game.initial_positions, game.env_config)
    env.reset()
    board_width = game.env_config.board_width
    total_cost = 0
    layer_move_tensors = []
    for action in game.history:
        qubit_idx = env.current_qubit
        current_flat = (env.atom_positions[qubit_idx][0] * board_width +
                        env.atom_positions[qubit_idx][1]).item()
        layer_before = env.tasks_done
        if action != current_flat:
            src = env.atom_positions[qubit_idx].tolist()
            dst_row, dst_col = action // board_width, action % board_width
            layer_move_tensors.append(torch.tensor([src[0], src[1], dst_row, dst_col], dtype=torch.long))
        env.step(action)
        if env.tasks_done > layer_before:
            # Layer just auto-executed — compute its cost
            if layer_move_tensors:
                total_cost += count_groups(torch.stack(layer_move_tensors), canonicalize=False)
            gate_moves = gates_to_moves(game.tasks[layer_before], env.atom_positions)
            if len(gate_moves) > 0:
                total_cost += 2 * count_groups(gate_moves, canonicalize=True)
            layer_move_tensors = []
    return total_cost


def game_to_solution(game):
    env = NeutralAtomsEnv(game.tasks, game.initial_positions, game.env_config)
    env.reset()
    board_width = game.env_config.board_width
    # plan[layer_idx] = list of moves for that layer (atom-viz does its own grouping)
    plan = [[] for _ in game.tasks]
    current_layer_moves = []

    for action in game.history:
        qubit_idx = env.current_qubit
        current_flat = (env.atom_positions[qubit_idx][0] * board_width +
                        env.atom_positions[qubit_idx][1]).item()
        layer_before = env.tasks_done

        if action != current_flat:
            src = env.atom_positions[qubit_idx].tolist()
            dst_row, dst_col = action // board_width, action % board_width
            current_layer_moves.append({
                'atom': qubit_idx,
                'from': {'row': src[0], 'col': src[1]},
                'to': {'row': dst_row, 'col': dst_col},
            })

        env.step(action)

        if env.tasks_done > layer_before:
            plan[layer_before] = current_layer_moves
            current_layer_moves = []

    initial_atoms = {
        str(i): {'row': r, 'col': c}
        for i, (r, c) in enumerate(game.initial_positions)
    }
    return {
        'board': {
            'rows': game.env_config.board_height,
            'cols': game.env_config.board_width,
            'initialAtoms': initial_atoms,
        },
        'circuit': game.tasks,
        'plan': plan,
    }


def save_solution(game, path):
    solution = game_to_solution(game)
    with open(path, 'w') as f:
        json.dump(solution, f, indent=2)
    return solution


def log_game_trace(game, run_dir, label='best'):
    env = NeutralAtomsEnv(game.tasks, game.initial_positions, game.env_config)
    env.reset()
    board_width = game.env_config.board_width
    trace = []
    for step_idx, action in enumerate(game.history):
        cost_before = env._get_total_cost()
        qubit_idx = env.current_qubit
        current_flat = (env.atom_positions[qubit_idx][0] * board_width +
                        env.atom_positions[qubit_idx][1]).item()
        layer_before = env.tasks_done
        entry = {'step': step_idx, 'action': action, 'qubit': qubit_idx, 'layer': layer_before}
        if action == current_flat:
            entry['action_type'] = 'noop'
        else:
            src = env.atom_positions[qubit_idx].tolist()
            entry['action_type'] = 'move'
            entry['src'] = src
            entry['dst'] = [action // board_width, action % board_width]
        result = env.step(action)
        entry['cost_before'] = cost_before
        entry['cost_after'] = env._get_total_cost()
        entry['reward'] = result.reward
        entry['tasks_done'] = env.tasks_done
        if env.tasks_done > layer_before:
            entry['layer_completed'] = layer_before
        trace.append(entry)
    path = os.path.join(run_dir, 'solutions', f'{label}_trace.json')
    with open(path, 'w') as f:
        json.dump(trace, f, indent=2)
    return trace


def selfplay_metrics(games):
    costs = [compute_solution_cost(g) for g in games]
    avg_cost = sum(costs) / len(costs)
    total_steps = sum(len(g.history) for g in games)
    total_moves = sum(g.last_info.get('num_moves', 0) for g in games)
    total_move_dist = sum(g.last_info.get('total_move_distance', 0) for g in games)
    metrics = {
        'best_cost': min(costs),
        'avg_cost': round(avg_cost, 1),
        'cost_std': round((sum((c - avg_cost)**2 for c in costs) / len(costs)) ** 0.5, 1),
        'noop_frac': round(1 - total_moves / total_steps, 2) if total_steps > 0 else 0,
        'avg_move_dist': round(total_move_dist / max(total_moves, 1), 1),
        'best_game': games[costs.index(min(costs))],
    }
    all_root_values = [v for g in games for v in g.root_values]
    if all_root_values:
        metrics['avg_root_value'] = sum(all_root_values) / len(all_root_values)
    all_entropies = []
    for g in games:
        for visits in g.child_visits:
            probs = [p for p in visits if p > 0]
            if probs:
                all_entropies.append(-sum(p * math.log(p) for p in probs))
    if all_entropies:
        metrics['avg_policy_entropy'] = sum(all_entropies) / len(all_entropies)
    # MCTS diagnostics
    all_depths = [d for g in games for d in g.mcts_depths]
    all_reward_fracs = [f for g in games for f in g.mcts_reward_fracs]
    all_reward_sum_means = [x for g in games for x in g.mcts_reward_sum_means]
    all_reward_sum_stds = [x for g in games for x in g.mcts_reward_sum_stds]
    all_reward_abs_sum_means = [x for g in games for x in g.mcts_reward_abs_sum_means]
    all_boundary_fracs = [x for g in games for x in g.mcts_boundary_reach_fracs]
    all_sign_changes = [x for g in games for x in g.mcts_sign_changes_means]
    if all_depths:
        metrics['avg_mcts_depth'] = round(sum(all_depths) / len(all_depths), 1)
    if all_reward_fracs:
        metrics['mcts_reward_frac'] = round(sum(all_reward_fracs) / len(all_reward_fracs), 2)
    if all_reward_sum_means:
        metrics['mcts_reward_sum_mean'] = round(sum(all_reward_sum_means) / len(all_reward_sum_means), 3)
    if all_reward_sum_stds:
        metrics['mcts_reward_sum_std'] = round(sum(all_reward_sum_stds) / len(all_reward_sum_stds), 3)
    if all_reward_abs_sum_means:
        metrics['mcts_reward_abs_sum_mean'] = round(sum(all_reward_abs_sum_means) / len(all_reward_abs_sum_means), 3)
    if all_boundary_fracs:
        metrics['mcts_boundary_reach_frac'] = round(sum(all_boundary_fracs) / len(all_boundary_fracs), 2)
    if all_sign_changes:
        metrics['mcts_sign_changes_mean'] = round(sum(all_sign_changes) / len(all_sign_changes), 2)
    # Reward stats
    all_rewards = [r for g in games for r in g.rewards]
    if all_rewards:
        metrics['avg_reward'] = round(sum(all_rewards) / len(all_rewards), 3)
        nonzero = [r for r in all_rewards if abs(r) > 1e-6]
        metrics['reward_nonzero_frac'] = round(len(nonzero) / len(all_rewards), 2)
    # Cost histogram: fraction of games at each cost bucket
    cost_counts = {}
    for c in costs:
        cost_counts[c] = cost_counts.get(c, 0) + 1
    metrics['cost_hist'] = {str(k): round(v / len(costs), 3) for k, v in sorted(cost_counts.items())}
    # Value calibration: correlation between per-game mean root_value (normalized by map's
    # do_nothing baseline) and actual cost. Normalization is essential for multi-map training:
    # raw root_value = do_nothing(map) - actual_cost under Option E, so maps with higher
    # do_nothing baselines inflate root_value independently of policy quality.
    # Normalized: adj_root_value = do_nothing - root_value ≈ expected_actual_cost, so
    # correlation with actual_cost should be positive when V is accurate.
    game_root_vals = [sum(g.root_values) / len(g.root_values) if g.root_values else None for g in games]
    do_nothings = [g.environment.cost_ub for g in games]
    pairs = [(dn - rv, c) for rv, dn, c in zip(game_root_vals, do_nothings, costs) if rv is not None]
    if len(pairs) >= 2:
        adj_rvs = [p[0] for p in pairs]
        cs = [p[1] for p in pairs]
        mean_rv, mean_c = sum(adj_rvs) / len(adj_rvs), sum(cs) / len(cs)
        cov = sum((r - mean_rv) * (c - mean_c) for r, c in pairs) / len(pairs)
        std_rv = (sum((r - mean_rv)**2 for r in adj_rvs) / len(adj_rvs)) ** 0.5
        std_c = (sum((c - mean_c)**2 for c in cs) / len(cs)) ** 0.5
        metrics['val_cost_corr'] = round(cov / (std_rv * std_c + 1e-8), 3)
    # Per-layer entropy: split episode steps by layer boundaries
    layer_entropies = {}
    for g in games:
        boundaries = [0]
        for atoms in g.environment.layer_relevant_atoms:
            boundaries.append(boundaries[-1] + len(atoms))
        for layer_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            key = f'entropy_layer{layer_idx}'
            for step in range(start, min(end, len(g.child_visits))):
                probs = [p for p in g.child_visits[step] if p > 0]
                if probs:
                    layer_entropies.setdefault(key, []).append(-sum(p * math.log(p) for p in probs))
    for key, vals in layer_entropies.items():
        metrics[key] = round(sum(vals) / len(vals), 3)
    return metrics


def append_epoch_metrics(run_dir, metrics):
    path = os.path.join(run_dir, 'metrics.jsonl')
    with open(path, 'a') as f:
        f.write(json.dumps(metrics) + '\n')


def append_to_registry(output_dir, run_id, config, metrics):
    entry = {
        'run_id': run_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'study': config.experiment.study,
        'hypothesis': config.experiment.hypothesis,
        'variant': config.experiment.variant,
        'tags': _split_tags(config.experiment.tags),
        'notes': config.experiment.notes,
        'decision': config.experiment.decision,
        'parent_run': config.experiment.parent_run,
        'map_num': config.map_num,
        'use_fake': config.use_fake,
        'random_board': config.random_board,
        'reward_mode': config.env.reward_mode,
        'num_simulations': config.mcts.num_simulations,
        'plan_cost_search_bonus_weight': config.mcts.plan_cost_search_bonus_weight,
        'root_dirichlet_alpha': config.mcts.root_dirichlet_alpha,
        'root_exploration_fraction': config.mcts.root_exploration_fraction,
        'temperature_init': config.mcts.temperature_init,
        'temperature_final': config.mcts.temperature_final,
        'temperature_decay_steps': config.mcts.temperature_decay_steps,
        'lr': config.training.lr,
        'batch_size': config.training.batch_size,
        'training_steps': config.training.training_steps,
        'num_selfplay': config.training.num_selfplay,
        'num_parallel_games': config.training.num_parallel_games,
        'seed': config.training.seed,
        'data_augmentation': config.training.data_augmentation,
        'fixed_map_fraction': config.training.fixed_map_fraction,
        'curriculum_maps': config.experiment.curriculum_maps,
        'curriculum_patience': config.experiment.curriculum_patience,
        'curriculum_initial_phase': config.experiment.curriculum_initial_phase,
        'load_checkpoint': config.experiment.load_checkpoint,
        'correctness_weight': config.network.correctness_weight,
        'latency_weight': config.network.latency_weight,
    }
    entry.update(_git_metadata())
    entry.update({k: v for k, v in metrics.items() if k != 'best_game'})
    registry_path = os.path.join(output_dir, 'run_registry.jsonl')
    with open(registry_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')

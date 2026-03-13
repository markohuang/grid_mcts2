import json
import math
import os
import hashlib
import time
import torch

from .env import NeutralAtomsEnv
from .tasks import gates_to_moves, get_relevant_atoms
from .moves import parallel_groups, count_groups


def create_run_dir(config):
    run_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
    run_dir = os.path.join(config.experiment.output_dir, run_id)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'solutions'), exist_ok=True)
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
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
    plan = []
    current_layer_moves = []
    current_layer_tensors = []
    prev_layer = 0

    for action in game.history:
        qubit_idx = env.current_qubit
        current_flat = (env.atom_positions[qubit_idx][0] * board_width +
                        env.atom_positions[qubit_idx][1]).item()
        layer_before = env.tasks_done

        if action != current_flat:
            src = env.atom_positions[qubit_idx].tolist()

        result = env.step(action)
        layer_after = env.tasks_done

        if action != current_flat:
            dst = env.atom_positions[qubit_idx].tolist() if layer_after == layer_before else src
            # If layer advanced, the atom was placed before auto-execute
            # Reconstruct dst from action
            dst_row, dst_col = action // board_width, action % board_width
            current_layer_moves.append((qubit_idx, src, [dst_row, dst_col]))
            move_tensor = torch.tensor([src[0], src[1], dst_row, dst_col], dtype=torch.long)
            current_layer_tensors.append(move_tensor)

        if layer_after > layer_before:
            # Layer auto-executed, flush moves
            plan.extend(_group_phase_moves(current_layer_moves, current_layer_tensors))
            current_layer_moves = []
            current_layer_tensors = []

    plan.extend(_group_phase_moves(current_layer_moves, current_layer_tensors))

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


def _group_phase_moves(phase_moves, phase_tensors):
    if not phase_moves:
        return []
    groups = parallel_groups(torch.stack(phase_tensors), canonicalize=True)
    result = []
    for g in range(groups.max().item() + 1):
        group_entries = []
        for i, (atom, src, dst) in enumerate(phase_moves):
            if groups[i].item() == g:
                group_entries.append({
                    'atom': atom,
                    'from': {'row': src[0], 'col': src[1]},
                    'to': {'row': dst[0], 'col': dst[1]},
                })
        result.append(group_entries)
    return result


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
        cost_before = env._cached_cost
        qubit_idx = env.current_qubit
        current_flat = (env.atom_positions[qubit_idx][0] * board_width +
                        env.atom_positions[qubit_idx][1]).item()
        entry = {'step': step_idx, 'action': action, 'qubit': qubit_idx}
        if action == current_flat:
            entry['action_type'] = 'noop'
        else:
            src = env.atom_positions[qubit_idx].tolist()
            entry['action_type'] = 'move'
            entry['src'] = src
            entry['dst'] = [action // board_width, action % board_width]
        layer_before = env.tasks_done
        result = env.step(action)
        entry['cost_before'] = cost_before
        entry['cost_after'] = env._cached_cost
        entry['reward'] = result.reward
        entry['tasks_done'] = env.tasks_done
        if env.tasks_done > layer_before:
            entry['layer_completed'] = layer_before
        trace.append(entry)
    path = os.path.join(run_dir, 'solutions', f'{label}_trace.json')
    with open(path, 'w') as f:
        json.dump(trace, f, indent=2)
    return trace


def selfplay_metrics(games, num_tasks):
    completed = [g for g in games if g.last_info.get('tasks_done', 0) >= num_tasks]
    metrics = {
        'completion_rate': len(completed) / len(games),
        'avg_steps': sum(len(g.history) for g in games) / len(games),
    }
    if completed:
        costs = [compute_solution_cost(g) for g in completed]
        metrics['best_cost'] = min(costs)
        metrics['avg_cost'] = sum(costs) / len(costs)
        best_idx = costs.index(min(costs))
        metrics['best_game'] = completed[best_idx]
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
    return metrics


def append_epoch_metrics(run_dir, metrics):
    path = os.path.join(run_dir, 'metrics.jsonl')
    with open(path, 'a') as f:
        f.write(json.dumps(metrics) + '\n')


def append_to_registry(output_dir, run_id, config, metrics):
    entry = {
        'run_id': run_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'map_num': config.map_num,
        'use_fake': config.use_fake,
        'num_simulations': config.mcts.num_simulations,
        'lr': config.training.lr,
        'batch_size': config.training.batch_size,
        'training_steps': config.training.training_steps,
        'num_selfplay': config.training.num_selfplay,
        'num_parallel_games': config.training.num_parallel_games,
        'correctness_weight': config.network.correctness_weight,
        'latency_weight': config.network.latency_weight,
    }
    entry.update({k: v for k, v in metrics.items() if k != 'best_game'})
    registry_path = os.path.join(output_dir, 'run_registry.jsonl')
    with open(registry_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')

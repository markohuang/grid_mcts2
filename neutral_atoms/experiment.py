import json
import os
import hashlib
import time
import torch

from .env import NeutralAtomsEnv
from .board import action_to_move
from .tasks import gates_to_moves
from .moves import parallel_groups, count_groups
from .types import GATE_ACTION


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
    board_shape = (game.env_config.board_height, game.env_config.board_width)
    total_cost = 0
    phase_move_tensors = []
    for action in game.history:
        if action == GATE_ACTION:
            if phase_move_tensors:
                total_cost += count_groups(torch.stack(phase_move_tensors), canonicalize=False)
            gate_moves = gates_to_moves(game.tasks[env.tasks_done], env.atom_positions)
            if len(gate_moves) > 0:
                total_cost += 2 * count_groups(gate_moves, canonicalize=True)
            phase_move_tensors = []
            env.step(action)
        else:
            move = action_to_move(action, env.atom_positions, board_shape)
            phase_move_tensors.append(move)
            env.step(action)
    if phase_move_tensors:
        total_cost += count_groups(torch.stack(phase_move_tensors), canonicalize=False)
    return total_cost


def game_to_solution(game):
    env = NeutralAtomsEnv(game.tasks, game.initial_positions, game.env_config)
    env.reset()
    board_shape = (game.env_config.board_height, game.env_config.board_width)
    board_size = board_shape[0] * board_shape[1]
    plan = []
    phase_moves, phase_tensors = [], []
    for action in game.history:
        if action == GATE_ACTION:
            plan.extend(_group_phase_moves(phase_moves, phase_tensors))
            phase_moves, phase_tensors = [], []
            env.step(action)
        else:
            q = (action - 1) // board_size
            src = env.atom_positions[q].tolist()
            move_tensor = action_to_move(action, env.atom_positions, board_shape)
            env.step(action)
            dst = env.atom_positions[q].tolist()
            phase_moves.append((q, src, dst))
            phase_tensors.append(move_tensor)
    plan.extend(_group_phase_moves(phase_moves, phase_tensors))
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
    return metrics


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
    }
    entry.update({k: v for k, v in metrics.items() if k != 'best_game'})
    registry_path = os.path.join(output_dir, 'run_registry.jsonl')
    with open(registry_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')

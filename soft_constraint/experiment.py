"""
experiment.py — Run tracking, atom-viz JSON output, metrics logging.

Mirrors the patterns from neutral_atoms/experiment.py adapted for
the soft constraint optimizer (no MCTS, no Game objects).
"""

import json
import os
import hashlib
import time
import torch


def create_run_dir(config):
    """Create run directory with config snapshot.

    Returns: (run_id, run_dir)
    """
    run_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
    run_dir = os.path.join(config.experiment.output_dir, run_id)
    os.makedirs(os.path.join(run_dir, 'solutions'), exist_ok=True)
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    return run_id, run_dir


def make_atom_viz_json(init_cells, layer_cells, tasks, H, W):
    """Build atom-viz compatible JSON from cell assignments.

    Args:
        init_cells: (N,) initial cell indices
        layer_cells: list of (N,) cell indices per layer
        tasks: list of gate lists per layer
        H, W: grid dims

    Returns: dict with board, circuit, plan keys
    """
    N = init_cells.shape[0]

    # Board: initial atom positions
    initial_atoms = {}
    for i in range(N):
        c = init_cells[i].item()
        initial_atoms[str(i)] = {'row': c // W, 'col': c % W}

    # Plan: moves per layer (only atoms that moved)
    plan = []
    prev = init_cells
    for t, cells_t in enumerate(layer_cells):
        layer_moves = []
        for q in range(N):
            src_c = prev[q].item()
            dst_c = cells_t[q].item()
            if src_c != dst_c:
                layer_moves.append({
                    'atom': q,
                    'from': {'row': src_c // W, 'col': src_c % W},
                    'to': {'row': dst_c // W, 'col': dst_c % W},
                })
        plan.append(layer_moves)
        prev = cells_t

    return {
        'board': {'rows': H, 'cols': W, 'initialAtoms': initial_atoms},
        'circuit': tasks,
        'plan': plan,
    }


def save_solution(run_dir, init_cells, layer_cells, tasks, H, W, label='best'):
    """Save atom-viz JSON to run_dir/solutions/<label>.json."""
    solution = make_atom_viz_json(init_cells, layer_cells, tasks, H, W)
    path = os.path.join(run_dir, 'solutions', f'{label}.json')
    with open(path, 'w') as f:
        json.dump(solution, f, indent=2)
    return solution


def append_metrics(run_dir, metrics):
    """Append a metrics dict as one JSON line to metrics.jsonl."""
    path = os.path.join(run_dir, 'metrics.jsonl')
    with open(path, 'a') as f:
        f.write(json.dumps(metrics) + '\n')


def append_to_registry(output_dir, run_id, config, metrics):
    """Append run summary to run_registry.jsonl."""
    entry = {
        'run_id': run_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'map_num': config.map_num,
        'lr': config.optimizer.lr,
        'n_steps': config.optimizer.n_steps,
        'n_restarts': config.optimizer.n_restarts,
        'lambda_g': config.surrogate.lambda_g,
        'lambda_r': config.surrogate.lambda_r,
        'lambda_aux': config.surrogate.lambda_aux,
    }
    entry.update(metrics)
    registry_path = os.path.join(output_dir, 'run_registry.jsonl')
    with open(registry_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')

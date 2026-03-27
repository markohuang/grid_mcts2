"""
main.py — Soft constraint optimizer entry point.

Usage:
    python main.py
    python main.py --config.optimizer.n_steps=10  # smoke test
    python main.py --config.surrogate.lambda_aux=1.0
    python main.py --config.map_num=1
"""

import time
import torch
import torch.nn.functional as F
from absl import app
from ml_collections import config_flags
from lightning import Fabric

from config import get_config, set_derived_config, get_map_data, atom_map_to_positions
from surrogate import (
    gate_cost_surrogate, reconfig_cost_surrogate, get_relevant_atoms,
    true_total_cost,
)
from experiment import create_run_dir, save_solution, append_metrics, append_to_registry


_CONFIG = config_flags.DEFINE_config_dict('config', get_config())


def build_init_state(map_data, H, W, C, N, device, dtype):
    """Build initial cell indices and one-hot distributions."""
    positions = atom_map_to_positions(map_data['atom_map'], W)
    pos_tensor = torch.tensor(positions, device=device, dtype=torch.long)
    init_cells = pos_tensor[:, 0] * W + pos_tensor[:, 1]  # (N,)
    init_dists = torch.zeros(N, C, device=device, dtype=dtype)
    init_dists.scatter_(1, init_cells.unsqueeze(1), 1.0)
    return init_cells, init_dists, pos_tensor


def init_logits(tasks, relevant, pos_tensor, W, C, config, device, dtype):
    """Create per-layer logits for relevant atoms, biased toward initial positions."""
    layer_logits = []
    for t in range(len(tasks)):
        n_rel = len(relevant[t])
        logits_t = torch.randn(n_rel, C, device=device, dtype=dtype) * config.optimizer.logit_init_scale
        for local_idx, global_idx in enumerate(relevant[t]):
            cell = (pos_tensor[global_idx, 0] * W + pos_tensor[global_idx, 1]).item()
            logits_t[local_idx, cell] += config.optimizer.logit_bias
        logits_t.requires_grad_(True)
        layer_logits.append(logits_t)
    return layer_logits


def build_layer_dists(layer_logits, init_dists, relevant, N):
    """Build per-layer (N, C) distributions from logits."""
    layer_dists = []
    prev_dist = init_dists
    for t in range(len(layer_logits)):
        dist_t = prev_dist.clone().detach()
        for local_idx, global_idx in enumerate(relevant[t]):
            dist_t[global_idx] = F.softmax(layer_logits[t][local_idx], dim=-1)
        layer_dists.append(dist_t)
        prev_dist = dist_t
    return layer_dists


def compute_loss(layer_dists, init_dists, tasks, relevant, H, W, config):
    """Compute total surrogate loss across all layers."""
    total_loss = torch.tensor(0.0, device=init_dists.device, dtype=init_dists.dtype)
    s = config.surrogate
    for t in range(len(tasks)):
        gc, _ = gate_cost_surrogate(
            tasks[t], layer_dists[t], H, W,
            lambda_g=s.lambda_g, lambda_aux=s.lambda_aux,
            loss_mode=s.loss_mode, delta=s.delta)
        src_d = init_dists if t == 0 else layer_dists[t - 1]
        rc, _ = reconfig_cost_surrogate(
            src_d, layer_dists[t], H, W,
            mover_indices=relevant[t],
            lambda_r=s.lambda_r, lambda_aux=s.lambda_aux,
            loss_mode=s.loss_mode, delta=s.delta)
        total_loss = total_loss + gc + rc
    return total_loss


def evaluate(layer_dists, init_cells, tasks, H, W):
    """Hard-assign and compute true cost."""
    hard_cells_list = []
    for t in range(len(tasks)):
        hard_cells_list.append(layer_dists[t].argmax(dim=-1))
    true_cost, breakdown = true_total_cost(init_cells, hard_cells_list, tasks, H, W)
    return true_cost, breakdown, hard_cells_list


def run_single(seed, tasks, relevant, init_cells, init_dists, pos_tensor,
               H, W, C, N, config, device, dtype):
    """Run a single restart and return best (cost, breakdown, hard_cells, history)."""
    torch.manual_seed(seed)
    layer_logits = init_logits(tasks, relevant, pos_tensor, W, C, config, device, dtype)
    optimizer = torch.optim.Adam(layer_logits, lr=config.optimizer.lr)

    best_cost, best_breakdown, best_cells = None, None, None
    history = []

    for step in range(config.optimizer.n_steps):
        optimizer.zero_grad()
        layer_dists = build_layer_dists(layer_logits, init_dists, relevant, N)
        loss = compute_loss(layer_dists, init_dists, tasks, relevant, H, W, config)
        loss.backward()
        optimizer.step()

        if step % config.experiment.eval_interval == 0 or step == config.optimizer.n_steps - 1:
            with torch.no_grad():
                true_cost, breakdown, hard_cells = evaluate(layer_dists, init_cells, tasks, H, W)
                history.append({
                    'step': step,
                    'surrogate': loss.item(),
                    'true_cost': true_cost,
                    'breakdown': breakdown,
                })
                if best_cost is None or true_cost < best_cost:
                    best_cost = true_cost
                    best_breakdown = breakdown
                    best_cells = [hc.clone() for hc in hard_cells]

    return best_cost, best_breakdown, best_cells, history


def main(_):
    config = _CONFIG.value
    set_derived_config(config)

    fabric = Fabric(accelerator=config.training.accelerator,
                    devices=config.training.devices)
    fabric.launch()
    device = fabric.device
    dtype = torch.float64

    # Map setup
    map_data = get_map_data(config)
    tasks = map_data['tasks']
    H, W = config.env.board_height, config.env.board_width
    C = config.env.board_size
    N = config.env.num_qubits
    relevant = [get_relevant_atoms(t) for t in tasks]

    init_cells, init_dists, pos_tensor = build_init_state(map_data, H, W, C, N, device, dtype)

    # Baseline
    baseline = sum(
        2 * len(t) for t in tasks  # upper bound: each gate in own group
    )
    from surrogate import true_gate_cost
    baseline = sum(true_gate_cost(init_cells, t, H, W) for t in tasks)

    # Run info
    print(f"Map {config.map_num}: {H}×{W}, {N} atoms, {len(tasks)} layers")
    print(f"Do-nothing baseline: {baseline}")
    print(f"Restarts: {config.optimizer.n_restarts}, Steps: {config.optimizer.n_steps}, "
          f"λ_g={config.surrogate.lambda_g}, λ_r={config.surrogate.lambda_r}, "
          f"λ_aux={config.surrogate.lambda_aux}")
    print(f"Device: {device}")

    # Create run dir
    run_id, run_dir = create_run_dir(config)
    print(f"Run: {run_id} → {run_dir}")

    # Multi-restart optimization
    global_best_cost = None
    global_best_breakdown = None
    global_best_cells = None
    all_costs = []

    t_start = time.time()
    for restart in range(config.optimizer.n_restarts):
        seed = config.experiment.seed + restart * 100
        cost, breakdown, cells, history = run_single(
            seed, tasks, relevant, init_cells, init_dists, pos_tensor,
            H, W, C, N, config, device, dtype)
        all_costs.append(cost)

        bd_str = ' '.join(f'r{r}+g{g}' for r, g in breakdown)
        print(f"  restart {restart}: cost={cost} [{bd_str}]")

        # Save per-restart solution (atom-viz JSON)
        save_solution(run_dir, init_cells, cells, tasks, H, W,
                      label=f'restart_{restart:03d}_cost{cost}')

        # Log per-restart metrics
        append_metrics(run_dir, {
            'restart': restart, 'seed': seed,
            'best_cost': cost, 'breakdown': breakdown,
            'history': history,
        })

        if global_best_cost is None or cost < global_best_cost:
            global_best_cost = cost
            global_best_breakdown = breakdown
            global_best_cells = cells

    elapsed = time.time() - t_start

    # Summary
    print(f"\nBest: {global_best_cost} (baseline: {baseline})")
    if global_best_breakdown:
        bd_str = ' '.join(f'r{r}+g{g}' for r, g in global_best_breakdown)
        print(f"  breakdown: {bd_str}")
    print(f"All costs: {all_costs}")
    print(f"Mean: {sum(all_costs)/len(all_costs):.1f}, Time: {elapsed:.1f}s")

    # Save best solution
    if global_best_cells:
        save_solution(run_dir, init_cells, global_best_cells, tasks, H, W)

    # Registry
    summary = {
        'best_cost': global_best_cost,
        'mean_cost': round(sum(all_costs) / len(all_costs), 1),
        'all_costs': all_costs,
        'baseline': baseline,
        'elapsed_s': round(elapsed, 1),
    }
    if global_best_breakdown:
        summary['breakdown'] = global_best_breakdown
    append_to_registry(config.experiment.output_dir, run_id, config, summary)
    print(f"\nSaved to {run_dir}")


if __name__ == '__main__':
    app.run(main)

import torch
import math
from .types import Board, AtomPositions, Tasks
from .moves import count_groups, group_sizes, is_parallel_executable_batch
from .tasks import gates_to_moves
from .board import apply_moves_batch

def compute_entropy(sizes: torch.Tensor) -> float:
    if len(sizes) == 0 or sizes.sum() == 0:
        return 0.0
    probs = sizes.float() / sizes.sum()
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    return entropy

def compute_normalized_entropy(sizes: torch.Tensor, num_moves: int) -> float:
    if num_moves <= 1:
        return 0.0
    max_ent = math.log(num_moves)
    return compute_entropy(sizes) / max_ent if max_ent > 0 else 0.0

def compute_total_cost(
    board: Board,
    atom_positions: AtomPositions,
    tasks: Tasks,
    tasks_done: int,
    current_phase_moves: list,
) -> tuple[int, float]:
    # returns (total_groups, avg_entropy)
    total_groups = 0
    total_entropy = 0.0
    entropy_count = 0
    sim_board = board.clone()
    sim_positions = atom_positions.clone()

    for layer_idx in range(tasks_done, len(tasks)):
        # reconfig cost (only current layer has planned moves)
        if layer_idx == tasks_done and len(current_phase_moves) > 0:
            reconfig_moves = torch.stack(current_phase_moves)
            total_groups += count_groups(reconfig_moves, canonicalize=False)
            sim_board, sim_positions = apply_moves_batch(sim_board, sim_positions, reconfig_moves)
            sizes = group_sizes(reconfig_moves, canonicalize=False)
            total_entropy += compute_normalized_entropy(sizes, len(reconfig_moves))
            entropy_count += 1
        # gate execution cost
        gate_moves = gates_to_moves(tasks[layer_idx], sim_positions)
        if len(gate_moves) > 0:
            total_groups += 2 * count_groups(gate_moves, canonicalize=True) # each gate costs 2 groups (enter + exit)
            sizes = group_sizes(gate_moves, canonicalize=True)
            total_entropy += compute_normalized_entropy(sizes, len(gate_moves))
            entropy_count += 1

    avg_entropy = total_entropy / entropy_count if entropy_count > 0 else 0.0
    return total_groups, avg_entropy

def compute_gate_only_cost(
    board: Board,
    atom_positions: AtomPositions,
    tasks: Tasks,
    tasks_done: int,
    current_phase_moves: list,
) -> int:
    sim_board = board.clone()
    sim_positions = atom_positions.clone()
    total_groups = 0
    for layer_idx in range(tasks_done, len(tasks)):
        if layer_idx == tasks_done and len(current_phase_moves) > 0:
            reconfig_moves = torch.stack(current_phase_moves)
            sim_board, sim_positions = apply_moves_batch(sim_board, sim_positions, reconfig_moves)
        gate_moves = gates_to_moves(tasks[layer_idx], sim_positions)
        if len(gate_moves) > 0:
            total_groups += 2 * count_groups(gate_moves, canonicalize=True)
    return total_groups

def compute_conflict_count(
    board: Board,
    atom_positions: AtomPositions,
    tasks: Tasks,
    tasks_done: int,
    current_phase_moves: list,
) -> int:
    sim_board = board.clone()
    sim_positions = atom_positions.clone()
    total_conflicts = 0
    for layer_idx in range(tasks_done, len(tasks)):
        if layer_idx == tasks_done and len(current_phase_moves) > 0:
            reconfig_moves = torch.stack(current_phase_moves)
            sim_board, sim_positions = apply_moves_batch(sim_board, sim_positions, reconfig_moves)
        gate_moves = gates_to_moves(tasks[layer_idx], sim_positions)
        if len(gate_moves) > 1:
            compat = is_parallel_executable_batch(gate_moves)
            # count upper triangle of incompatible pairs
            n = compat.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    if not compat[i, j]:
                        total_conflicts += 1
    return total_conflicts

def compute_manhattan_cost(
    board: Board,
    atom_positions: AtomPositions,
    tasks: Tasks,
    tasks_done: int,
    current_phase_moves: list,
) -> int:
    sim_board = board.clone()
    sim_positions = atom_positions.clone()
    total_dist = 0
    for layer_idx in range(tasks_done, len(tasks)):
        if layer_idx == tasks_done and len(current_phase_moves) > 0:
            reconfig_moves = torch.stack(current_phase_moves)
            sim_board, sim_positions = apply_moves_batch(sim_board, sim_positions, reconfig_moves)
        for q1, q2 in tasks[layer_idx]:
            p1, p2 = sim_positions[q1], sim_positions[q2]
            total_dist += abs(p1[0] - p2[0]).item() + abs(p1[1] - p2[1]).item()
    return total_dist

REWARD_COST_FN = {
    'gate_only': compute_gate_only_cost,
    'conflict_count': compute_conflict_count,
    'manhattan': compute_manhattan_cost,
}

def compute_reward(
    prev_cost: int, curr_cost: int,
    prev_entropy: float, curr_entropy: float,
    entropy_weight: float, reward_scale: float = 1.0,
) -> float:
    cost_delta = prev_cost - curr_cost
    entropy_delta = prev_entropy - curr_entropy
    return reward_scale * (cost_delta + entropy_weight * entropy_delta)

def compute_cost_bounds(tasks: Tasks) -> tuple[int, int]:
    cost_lb = 2*len(tasks)  # best: 1 group per layer, but 2x for round trip
    cost_ub = 2*sum(len(layer) for layer in tasks)  # worst: each gate separate, 2x
    return cost_lb, cost_ub

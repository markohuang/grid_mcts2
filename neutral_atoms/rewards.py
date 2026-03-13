import torch
from .types import Board, AtomPositions, Tasks
from .moves import count_groups
from .tasks import gates_to_moves
from .board import apply_moves_batch

def compute_total_cost(
    board: Board,
    atom_positions: AtomPositions,
    tasks: Tasks,
    tasks_done: int,
    current_phase_moves: list,
) -> int:
    total_groups = 0
    sim_board = board.clone()
    sim_positions = atom_positions.clone()

    for layer_idx in range(tasks_done, len(tasks)):
        if layer_idx == tasks_done and len(current_phase_moves) > 0:
            reconfig_moves = torch.stack(current_phase_moves)
            total_groups += count_groups(reconfig_moves, canonicalize=False)
            sim_board, sim_positions = apply_moves_batch(sim_board, sim_positions, reconfig_moves)
        gate_moves = gates_to_moves(tasks[layer_idx], sim_positions)
        if len(gate_moves) > 0:
            total_groups += 2 * count_groups(gate_moves, canonicalize=True)

    return total_groups

def compute_current_layer_cost(
    atom_positions: AtomPositions,
    tasks: Tasks,
    tasks_done: int,
    current_phase_moves: list,
) -> int:
    if tasks_done >= len(tasks):
        return 0
    total = 0
    if len(current_phase_moves) > 0:
        reconfig_moves = torch.stack(current_phase_moves)
        total += count_groups(reconfig_moves, canonicalize=False)
    # atom_positions already reflects the reconfig moves applied so far
    gate_moves = gates_to_moves(tasks[tasks_done], atom_positions)
    if len(gate_moves) > 0:
        total += 2 * count_groups(gate_moves, canonicalize=True)
    return total

def compute_reward(prev_cost: int, curr_cost: int, reward_scale: float = 1.0) -> float:
    return reward_scale * (prev_cost - curr_cost)

def compute_cost_bounds(tasks: Tasks) -> tuple[int, int]:
    cost_lb = 2 * len(tasks)
    cost_ub = 2 * sum(len(layer) for layer in tasks)
    return cost_lb, cost_ub

import torch
from .types import GateLayer, Tasks, AtomPositions, Moves

def gates_to_moves(gate_layer: GateLayer, atom_positions: AtomPositions) -> Moves:
    if len(gate_layer) == 0:
        return torch.empty(0, 4, dtype=torch.long)
    moves = []
    for q1, q2 in gate_layer:
        move = torch.cat([atom_positions[q1], atom_positions[q2]])
        moves.append(move)
    return torch.stack(moves)

def get_remaining_tasks(tasks: Tasks, tasks_done: int) -> Tasks:
    return tasks[tasks_done:]

def count_total_gates(tasks: Tasks) -> int:
    return sum(len(layer) for layer in tasks)

def is_episode_done(tasks_done: int, num_tasks: int) -> bool:
    return tasks_done >= num_tasks
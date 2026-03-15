import random
import torch
from .types import Tasks


def generate_random_map(board_dim: tuple[int, int], num_qubits: int, num_layers: int,
                        gates_per_layer: int = None, seed: int = None) -> dict:
    if seed is not None:
        random.seed(seed)
    board_h, board_w = board_dim
    board_size = board_h * board_w
    assert num_qubits <= board_size, f"Can't place {num_qubits} qubits on {board_h}x{board_w} board"
    # Random initial placement
    all_cells = list(range(board_size))
    random.shuffle(all_cells)
    atom_map = all_cells[:num_qubits]
    # Random tasks: each layer is a random matching on the qubit set
    if gates_per_layer is None:
        gates_per_layer = num_qubits // 2
    tasks = []
    for _ in range(num_layers):
        tasks.append(_random_matching(num_qubits, gates_per_layer))
    return {
        'board_dim': board_dim,
        'num_qubits': num_qubits,
        'atom_map': atom_map,
        'tasks': tasks,
    }


def _random_matching(num_qubits: int, num_pairs: int) -> list[list[int]]:
    qubits = list(range(num_qubits))
    random.shuffle(qubits)
    pairs = []
    for i in range(0, min(2 * num_pairs, len(qubits)) - 1, 2):
        pairs.append([qubits[i], qubits[i + 1]])
    return pairs

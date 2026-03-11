import torch
from .types import Board, AtomPositions, Move, Moves, EMPTY_CELL

def create_board(height: int, width: int, initial_positions: list[tuple[int, int]]) -> tuple[Board, AtomPositions]:
    board = torch.full((height, width), EMPTY_CELL, dtype=torch.long)
    atom_positions = torch.zeros((len(initial_positions), 2), dtype=torch.long)
    for qubit_idx, (row, col) in enumerate(initial_positions):
        board[row, col] = qubit_idx
        atom_positions[qubit_idx] = torch.tensor([row, col])
    return board, atom_positions

def apply_move(board: Board, atom_positions: AtomPositions, move: Move) -> tuple[Board, AtomPositions]:
    src_row, src_col, dst_row, dst_col = move.tolist()
    qubit_idx = board[src_row, src_col].item()
    board = board.clone()
    atom_positions = atom_positions.clone()
    board[src_row, src_col] = EMPTY_CELL
    board[dst_row, dst_col] = qubit_idx
    atom_positions[qubit_idx] = torch.tensor([dst_row, dst_col])
    return board, atom_positions

def apply_moves_batch(board: Board, atom_positions: AtomPositions, moves: Moves) -> tuple[Board, AtomPositions]:
    for i in range(len(moves)):
        board, atom_positions = apply_move(board, atom_positions, moves[i])
    return board, atom_positions

def action_to_move(action: int, atom_positions: AtomPositions, board_shape: tuple[int, int]) -> Move:
    # action = 1 + qubit_idx * board_size + flat_dest (action 0 = GATE)
    action -= 1
    board_size = board_shape[0] * board_shape[1]
    qubit_idx = action // board_size
    flat_dest = action % board_size
    dst_row, dst_col = flat_dest // board_shape[1], flat_dest % board_shape[1]
    src_row, src_col = atom_positions[qubit_idx].tolist()
    return torch.tensor([src_row, src_col, dst_row, dst_col], dtype=torch.long)

def get_legal_actions(board: Board, atom_positions: AtomPositions, num_qubits: int, include_gate: bool = True) -> list[int]:
    actions = [0] if include_gate else []
    board_h, board_w = board.shape
    board_size = board_h * board_w
    empty_mask = (board == EMPTY_CELL).view(-1)
    for qubit_idx in range(num_qubits):
        row, col = atom_positions[qubit_idx].tolist()
        current_flat = row * board_w + col
        for flat_dest in range(board_size):
            if flat_dest != current_flat and empty_mask[flat_dest]:
                actions.append(1 + qubit_idx * board_size + flat_dest)
    return actions
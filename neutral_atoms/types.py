import torch
from dataclasses import dataclass

Board = torch.Tensor         # (H, W) int64, -1=empty, 0..N-1=qubit idx
AtomPositions = torch.Tensor # (N, 2) int64, (row, col) per qubit
Move = torch.Tensor          # (4,) int64, [src_row, src_col, dst_row, dst_col]
Moves = torch.Tensor         # (M, 4) int64, batch of moves
GateLayer = list[tuple[int, int]]
Tasks = list[GateLayer]

GATE_ACTION = 0
EMPTY_CELL = -1

@dataclass
class StepResult:
    observation: dict
    reward: float
    done: bool
    info: dict

def to_json(tensor: torch.Tensor) -> list:
    return tensor.tolist()

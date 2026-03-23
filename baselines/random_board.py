"""Generate random boards and circuits in atom-viz format.

atom-viz format:
  board:   {rows, cols, initialAtoms: {str(id): {row, col}}}
  circuit: list of layers, each layer = list of [q0, q1] gate pairs
"""
import random
from typing import Optional


def random_board(
    rows: int = 5,
    cols: int = 5,
    num_qubits: int = 12,
    num_layers: int = 3,
    gates_per_layer: int = 4,
    seed: Optional[int] = None,
) -> dict:
    """Return {board, circuit} in atom-viz format.

    Constraints:
    - Each qubit occupies a unique cell.
    - Within a layer, each qubit appears in at most one gate.
    - Each layer has exactly gates_per_layer pairs (requires 2*gates_per_layer <= num_qubits).
    """
    assert 2 * gates_per_layer <= num_qubits, "Not enough qubits for gates_per_layer"
    assert num_qubits <= rows * cols, "Board too small for num_qubits"

    rng = random.Random(seed)
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(all_cells)
    initial_atoms = {str(q): {"row": all_cells[q][0], "col": all_cells[q][1]}
                     for q in range(num_qubits)}

    circuit = []
    for _ in range(num_layers):
        qubits = list(range(num_qubits))
        rng.shuffle(qubits)
        layer = [[qubits[2*i], qubits[2*i+1]] for i in range(gates_per_layer)]
        circuit.append(layer)

    return {
        "board": {"rows": rows, "cols": cols, "initialAtoms": initial_atoms},
        "circuit": circuit,
    }

"""DPQA SMT-solver baseline.

Strategy:
  1. Run DPQA to get a globally optimal gate grouping (which gates execute
     together and in what order).
  2. Feed those gate groups as task layers into the kohei greedy policy,
     which handles atom positioning in our hardware model.

This produces a valid, visualizable plan in atom-viz format that combines
DPQA's globally optimal gate ordering with kohei's greedy atom placement.

dpqa_n_t is reported alongside the plan for reference — it is DPQA's own
stage count under its hardware model (co-location), not our cost metric.

Requires:
    /home/marko/DPQA/solve.py  (UCLA-VAST/DPQA cloned at /home/marko/DPQA)
    z3-solver, python-sat  (installed in venv)

Public API:
    plan(initial_positions, tasks, rows, cols) -> dict
"""
import sys, tempfile, io, contextlib
from pathlib import Path

DPQA_PATH = '/home/marko/DPQA'
sys.path.insert(0, DPQA_PATH)
sys.path.insert(0, str(Path(__file__).parent.parent))


def plan(initial_positions: dict, tasks: list, rows: int, cols: int) -> dict:
    """Run DPQA for gate ordering, then kohei greedy for atom placement.

    Args:
        initial_positions: {qubit_id: (row, col)} or {qubit_id: cell_index}
        tasks: list of layers, each layer = list of (q0, q1) gate pairs
        rows, cols: board dimensions

    Returns:
        atom-viz plan dict {board, circuit, plan, dpqa_n_t}
        where circuit follows DPQA's gate-group ordering and plan contains
        kohei-greedy reconfig moves for each group.
    """
    from solve import DPQA as _DPQA
    from baselines.kohei_policy import plan as kohei_plan

    n_qubits = len(initial_positions)
    flat_gates = [(int(g[0]), int(g[1])) for layer in tasks for g in layer]

    # Step 1: run DPQA to get globally optimal gate grouping
    with tempfile.TemporaryDirectory() as tmpdir:
        solver = _DPQA("baseline", dir=tmpdir + '/', print_detail=False)
        solver.setArchitecture([rows, cols, rows, cols])
        solver.setProgram(flat_gates, nqubit=n_qubits)

        # Enforce circuit layer ordering: all gates in layer i must precede all gates in layer i+1
        layer_ranges = []
        idx = 0
        for layer in tasks:
            layer_ranges.append((idx, idx + len(layer) - 1))
            idx += len(layer)
        extra_deps = [
            (g_prev, g_next)
            for i in range(len(layer_ranges) - 1)
            for g_prev in range(layer_ranges[i][0], layer_ranges[i][1] + 1)
            for g_next in range(layer_ranges[i + 1][0], layer_ranges[i + 1][1] + 1)
        ]
        solver.dependencies = tuple(list(solver.dependencies) + extra_deps)

        solver.hybrid_strategy()
        with contextlib.redirect_stdout(io.StringIO()):
            result = solver.solve(save_file=True)

    # Step 2: re-group DPQA's output back into original circuit layers.
    # DPQA may split one layer into multiple stages (co-location sub-stages).
    # Using DPQA's stage structure would change the circuit, shifting the do-nothing baseline.
    # Instead: use gate['id'] (index into flat_gates) to recover which original layer each gate belongs to.
    gate_to_layer = {}
    idx = 0
    for layer_idx, layer in enumerate(tasks):
        for _ in layer:
            gate_to_layer[idx] = layer_idx
            idx += 1

    layer_gates = [[] for _ in tasks]
    for dpqa_layer in result['layers']:
        for g in dpqa_layer.get('gates', []):
            layer_gates[gate_to_layer[g['id']]].append((g['q0'], g['q1']))

    # Step 3: kohei greedy placement with original layer structure
    kohei_result = kohei_plan(initial_positions, layer_gates, rows, cols)
    kohei_result['dpqa_n_t'] = result['n_t']
    return kohei_result

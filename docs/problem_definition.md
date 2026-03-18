# Neutral Atom Reconfiguration — Problem Definition

## Hardware Context

A grid of optical tweezers (H × W) holds neutral atoms. Each atom is a qubit. To execute a two-qubit gate, the two atoms must be brought close together via Rydberg interaction, then returned to their positions.

Atoms are moved by **AOD (Acousto-Optic Deflector) lasers** that operate on entire rows/columns simultaneously. This creates a fundamental constraint: two atom movements can only execute **in parallel** if their displacement vectors don't "cross" in either axis.

## The Board

```
Board: (H, W) integer grid
  -1 = empty cell
  0..N-1 = qubit index occupying that cell

atom_positions: (N, 2) tensor
  atom_positions[q] = (row, col) of qubit q

Example — Map 2 (5×5, 12 qubits):
  . 9 . . 0      atom_map = [4, 12, 7, 11, 15, 5, 2, 21, 22, 23, 20, 8]
  5 . 2 11 .      positions: q0=(0,4), q1=(2,2), q2=(1,2), q3=(2,1),
  6 3 1 . .                  q4=(3,0), q5=(1,0), q6=(0,2), q7=(4,1),
  4 . . . .                  q8=(4,2), q9=(4,3), q10=(4,0), q11=(1,3)
 10 7 8 9 .
```

## Tasks (Circuit Layers)

A quantum circuit is decomposed into **gate layers**. Each layer is a list of two-qubit gate pairs:

```python
tasks = [
    [[4, 3], [6, 7], [9, 8], [10, 11]],   # layer 0: 4 gate pairs
    [[2, 1], [4, 5], [7, 6], [8, 9]],      # layer 1: 4 gate pairs
    [[0, 1], [2, 3], [6, 5], [9, 8]],      # layer 2: 4 gate pairs
]
```

Gate `[q1, q2]` means qubits q1 and q2 must interact. The interaction requires bringing them together (enter), executing the gate, then separating them (exit).

## The AOD Constraint — Parallel Move Compatibility

A move is a 4-tuple `[src_row, src_col, dst_row, dst_col]`.

Two moves (i, j) can execute in **parallel** if and only if ALL three conditions hold:

```python
# Pairwise vectors between the two moves
v1 = (src_i - src_j)   # difference in source positions
v2 = (dst_i - dst_j)   # difference in destination positions

# Condition 1: No crossing in rows
row_ok = (both_sources_same_row AND both_dests_same_row)
      OR (neither_shares_row AND same_row_direction)
# i.e., if one pair shares a row but the other doesn't → CONFLICT

# Condition 2: No crossing in columns (same logic, y-axis)
col_ok = (both_sources_same_col AND both_dests_same_col)
      OR (neither_shares_col AND same_col_direction)

# Condition 3: No collision (different destinations)
no_collision = (dst_i != dst_j)

compatible = row_ok AND col_ok AND no_collision
```

Exact implementation:

```python
def is_parallel_executable_batch(moves):
    """Returns (N, N) bool tensor, True = can run in parallel."""
    frm, to = moves[:, :2], moves[:, 2:]
    v1 = frm.unsqueeze(0) - frm.unsqueeze(1)  # (N, N, 2)
    v2 = to.unsqueeze(0) - to.unsqueeze(1)
    v1_x, v1_y = v1[..., 0], v1[..., 1]
    v2_x, v2_y = v2[..., 0], v2[..., 1]

    # horizontal: no crossing in x
    both_same_col = (v1_x == 0) & (v2_x == 0)
    same_x_dir = torch.sign(v1_x) == torch.sign(v2_x)
    either_same_col = (v1_x == 0) | (v2_x == 0)
    h_ok = both_same_col | (~either_same_col & same_x_dir)

    # vertical: no crossing in y
    both_same_row = (v1_y == 0) & (v2_y == 0)
    same_y_dir = torch.sign(v1_y) == torch.sign(v2_y)
    either_same_row = (v1_y == 0) | (v2_y == 0)
    v_ok = both_same_row | (~either_same_row & same_y_dir)

    # collision: same destination
    no_collision = ~((v2_x == 0) & (v2_y == 0))

    result = h_ok & v_ok & no_collision
    result.fill_diagonal_(True)
    return result
```

## Parallel Grouping

Given a set of moves, we need to find the minimum number of **sequential groups** such that all moves within a group are pairwise parallel-compatible. This is a **graph coloring** problem on the conflict graph.

```python
def greedy_color_from_adjacency(can_parallel):
    """Greedy graph coloring. Returns (N,) group assignments."""
    conflicts = ~can_parallel
    conflicts.fill_diagonal_(False)
    colors = torch.full((N,), -1, dtype=torch.long)
    order = torch.argsort(conflicts.sum(dim=1), descending=True)  # most-conflicted first
    for idx in order:
        neighbor_colors = colors[conflicts[idx]]
        # Assign smallest color not used by neighbors
        colors[idx] = first_unused_color(neighbor_colors)
    return colors

count_groups(moves) = max(parallel_groups(moves)) + 1
```

Note: greedy coloring is not optimal but is fast and close (~0.2% suboptimal for n=3, ~1% for n=4).

## Gate Move Canonicalization

For a gate `[q1, q2]`, the "gate move" represents the spatial relationship: `[pos_q1 → pos_q2]`. But the gate is symmetric — either atom could move to the other. **Canonicalization** chooses the direction for each gate move to minimize the total number of parallel groups.

```python
def canonicalize_moves(moves):
    """For each move, try forward and reversed direction.
    Greedily pick the direction that minimizes pairwise conflicts."""
    flipped = [dst, src] for each [src, dst]
    # For each move i, count conflicts with already-canonicalized moves j<i
    # in both forward and reversed directions. Pick the one with fewer conflicts.
```

Canonicalization applies to **gate moves only** (direction is flexible). Reconfig moves have fixed direction (atom moves from current position to chosen cell).

## Cost Computation

The total cost of a solution is the sum of parallel groups across all layers:

```python
def compute_solution_cost(game):
    total_cost = 0
    for each layer:
        # 1. Reconfig cost: how many sequential AOD operations
        #    to move atoms to their chosen positions
        if layer has reconfig moves:
            total_cost += count_groups(reconfig_moves, canonicalize=False)

        # 2. Gate execution cost: how many sequential AOD operations
        #    to bring gate pairs together and return them
        #    Factor of 2: enter (bring together) + exit (separate)
        gate_moves = [pos_q1 → pos_q2 for each gate (q1,q2) in layer]
        total_cost += 2 * count_groups(gate_moves, canonicalize=True)

    return total_cost
```

Exact implementation:

```python
def compute_solution_cost(game):
    env = NeutralAtomsEnv(game.tasks, game.initial_positions, game.env_config)
    env.reset()
    board_width = game.env_config.board_width
    total_cost = 0
    layer_move_tensors = []
    for action in game.history:
        qubit_idx = env.current_qubit
        current_flat = (env.atom_positions[qubit_idx][0] * board_width +
                        env.atom_positions[qubit_idx][1]).item()
        layer_before = env.tasks_done
        if action != current_flat:
            src = env.atom_positions[qubit_idx].tolist()
            dst_row, dst_col = action // board_width, action % board_width
            layer_move_tensors.append(
                torch.tensor([src[0], src[1], dst_row, dst_col], dtype=torch.long))
        env.step(action)
        if env.tasks_done > layer_before:
            if layer_move_tensors:
                total_cost += count_groups(
                    torch.stack(layer_move_tensors), canonicalize=False)
            gate_moves = gates_to_moves(
                game.tasks[layer_before], env.atom_positions)
            if len(gate_moves) > 0:
                total_cost += 2 * count_groups(gate_moves, canonicalize=True)
            layer_move_tensors = []
    return total_cost
```

## Objective

**Minimize total cost** = total number of sequential AOD operations across all layers.

```
total_cost = Σ_layers (reconfig_groups + 2 × gate_groups)
```

### Baselines

- **Do-nothing** (all atoms stay in place): no reconfig cost, but gate groups may be high because atoms aren't well-positioned. This is the baseline to beat.
- **Lower bound**: 2 × num_layers (every layer needs at least 1 gate group × 2 for round-trip). Only achievable if all gates in every layer can execute in a single parallel group from the given positions.

### Example — Map 2

```
Do-nothing cost: 18 (3 layers × 3 gate groups × 2)
Lower bound: 6 (3 layers × 1 gate group × 2)
Best found: 12 (with MCTS + learning)
```

Layer 0 from initial positions has gates with vectors `(-1,1), (4,-1), (0,-1), (-3,3)` — these point in different directions, requiring 3 parallel groups. Good reconfiguration could align them to reduce groups, but each reconfig move itself costs at least 1 group.

## Key Difficulty

The optimization has a **coordination problem**: individual atom moves typically have negative or zero immediate benefit (they add reconfig cost without reducing gate cost). Only coordinated multi-atom placements reduce the gate group count enough to offset the reconfig cost. This makes the problem hard for greedy/local search methods — the agent must plan multiple moves ahead to achieve net benefit.

## Current MDP Formulation

The problem is formulated as a sequential decision process:

```
For each gate layer t = 0..num_layers-1:
    relevant_atoms = sorted({q for (q1,q2) in tasks[t] for q in [q1,q2]})
    For each atom q in relevant_atoms (fixed order):
        Agent chooses action = cell index (0..board_size-1)
        Legal actions: current cell (no-op) + all empty cells
    Layer t auto-executes after all atoms are placed

Episode length = Σ_t |relevant_atoms_t|  (deterministic)
Action space = board_size (e.g., 25 for 5×5, 64 for 8×8)
```

### Dense Reward

```python
reward = prev_current_layer_cost - curr_current_layer_cost
```

Where `current_layer_cost = reconfig_groups(moves so far) + 2 × gate_groups(current layer from current positions)`. Computed before auto-execute, using only the current layer (not all remaining layers).

This provides per-step signal: moves that improve the current layer's parallel execution cost get positive reward, moves that worsen it get negative reward.

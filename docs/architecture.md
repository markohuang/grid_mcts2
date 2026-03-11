# Neutral Atoms MCTS — Architecture

## The Hardware

A grid of optical tweezers holds atoms. To execute a two-qubit gate, two atoms must be brought close together (Rydberg interaction), then returned to their positions.

**Constraint:** Atoms are moved by AOD lasers that operate on entire rows/columns. Two moves can only execute in parallel if they don't "cross" each other (same relative direction in both x and y).

## The Optimization Goal

**Given:**

- A board with N atoms at initial positions
- A sequence of gate layers, each containing multiple two-qubit gates

**Find:** A sequence of reconfiguration moves that minimizes total parallel execution steps.

Each layer requires:

1. **Reconfig phase**: Move atoms into good positions (cost = parallel groups needed)
2. **Gate phase**: Bring pairs together, execute gate, return apart (cost = 2 x parallel groups)

**Best case:** All gates in a layer execute in 1 parallel group -> cost = 2 per layer

**Worst case:** Each gate sequential -> cost = 2 x num_gates per layer

## Core Logic

### State

```
board: (H, W) tensor          # -1 = empty, 0..N-1 = qubit index
atom_positions: (N, 2) tensor # quick lookup of each qubit's (row, col)
tasks_done: int               # completed gate layers
current_phase_moves: list     # reconfig moves planned for current layer
```

### Actions

- `GATE_ACTION (0)`: Execute current gate layer, advance to next
- `move (1+)`: Move qubit q to cell p, encoded as `1 + q * board_size + p`
- Any qubit can move to any empty cell (not just adjacent)
- Action space size: `1 + num_qubits * board_size`

### Parallel Grouping (AOD Constraint)

Two moves are represented as `[src_row, src_col, dst_row, dst_col]`.

For a pair of moves (i, j), compute:
- `v1_x, v1_y` = difference in source positions (row, col)
- `v2_x, v2_y` = difference in destination positions (row, col)

Two moves can parallelize iff ALL three conditions hold:

1. **No crossing in rows:**
   - `(v1_x == 0 AND v2_x == 0)` — both sources same row AND both destinations same row, OR
   - `(v1_x != 0 AND v2_x != 0 AND sign(v1_x) == sign(v2_x))` — neither shares a row, and same relative row-direction

   (If one pair shares a row but the other doesn't, it's NOT safe — ambiguous crossing.)

2. **No crossing in columns:** (same logic, y-axis)
   - `(v1_y == 0 AND v2_y == 0)`, OR
   - `(v1_y != 0 AND v2_y != 0 AND sign(v1_y) == sign(v2_y))`

3. **No collision:** Different destinations (`v2_x != 0 OR v2_y != 0`)

Grouping uses greedy graph coloring on the conflict graph (sorted by degree, descending).

**Gate move canonicalization:** For gate pairs, either atom can move to the other. `canonicalize_moves` greedily picks directions that minimize pairwise conflicts to reduce total parallel groups.

### Cost Computation

```
total_cost = 0
for each remaining layer (tasks_done .. num_tasks):
    if current layer and has planned reconfig moves:
        reconfig_cost = count_parallel_groups(reconfig_moves)
    gate_cost = 2 * count_parallel_groups(gate_interactions)  # 2x for round trip
    total_cost += reconfig_cost + gate_cost
```

### Reward

```
reward = reward_scale * (
    (prev_cost - curr_cost)                              # positive = fewer parallel steps
  + entropy_weight * (prev_entropy - curr_entropy)       # optional shaping
)
```

Computed every step by asking "what if we executed all remaining gates now?" This gives the agent a dense signal — every move that improves future parallelism is immediately rewarded.

### Episode Termination

Episode ends when:
- All gate layers executed (`tasks_done == num_tasks`), OR
- Action budget exhausted (`len(actions) >= budget`)

## File Structure

```
neutral_atoms/
├── __init__.py    # Package init
├── types.py       # Type aliases (Board, AtomPositions, Move, etc.), constants, EnvConfig
├── config.py      # MCTSConfig, TrainingConfig, NetworkConfig dataclasses
├── board.py       # Board creation, move application, action encoding/decoding
├── moves.py       # Vectorized parallel grouping, canonicalization, graph coloring
├── tasks.py       # Gate layer utilities (gates_to_moves, is_episode_done)
├── rewards.py     # Cost computation (parallel groups + entropy), reward function
├── env.py         # NeutralAtomsEnv: step, reset, clone, legal_actions, observation
├── game.py        # Game: episode wrapper with history, targets, search stats
├── mcts.py        # MCTS: tree search, UCB, expansion, backprop, action selection
├── network.py     # MLPMixer value/policy nets, EMA, FakeNet
├── trainer.py     # AlphaAtomsTrainer: self-play loop, replay buffer, training
└── test_env.py    # Tests
```

## Key Hyperparameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `num_simulations` | 50 | MCTS simulations per decision |
| `num_selfplay` | 20 | Games per epoch |
| `buffer_size` | 1000 | Replay buffer capacity |
| `batch_size` | 128 | Training batch size |
| `training_steps` | 200 | Gradient steps per epoch |
| `td_steps` | 5 | TD bootstrap horizon |
| `discount` | 1.0 | MCTS discount factor |
| `budget` | 24 | Max actions per episode |
| `num_bins` | 51 | Value distribution bins |
| `ema_decay` | 0.995 | Target network EMA |
| `pb_c_base/init` | 19652 / 1.25 | UCB exploration constants |

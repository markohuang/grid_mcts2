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

## Device Lifecycle

1. Network starts on CPU. First `trainer.fit()` call sets up Lightning Fabric once via `_setup_fabric()`, moving the model to GPU via `fabric.setup()`.
2. Fabric, wrapped model, and optimizer are stored as trainer instance state and reused across epochs.
3. During self-play, the network stays on its device. `Network.inference()` moves CPU observations to the model's device automatically.
4. EMA shadow parameters (target network) are synced to device via `EMA.to(device)` after `fabric.setup()`.
5. Environment always runs on CPU — this is correct and intentional.

## File Structure

```
main.py                    # Entry point: config loading, epoch loop, early stopping
neutral_atoms/
├── __init__.py            # Package exports
├── types.py               # Type aliases (Board, AtomPositions, Move, etc.), StepResult, constants
├── config.py              # ml_collections ConfigDict (get_config, set_derived_config), MAPS
├── board.py               # Board creation, move application, action encoding/decoding
├── moves.py               # Vectorized parallel grouping, canonicalization, graph coloring
├── tasks.py               # Gate layer utilities (gates_to_moves, is_episode_done)
├── rewards.py             # Cost computation (parallel groups + entropy), reward function
├── env.py                 # NeutralAtomsEnv: step, reset, clone, legal_actions, observation
├── game.py                # Game: episode wrapper with history, targets, search stats
├── mcts.py                # MCTS: tree search, UCB, expansion, backprop, action selection
├── network.py             # MLPMixer value/policy nets, EMA, FakeNet, make_features
├── trainer.py             # AlphaAtomsTrainer: self-play, replay buffer, Fabric training
├── experiment.py          # Run tracking: directories, solution export, cost metrics, registry
└── test_env.py            # Tests for parallel grouping / move canonicalization
```

## Key Interfaces

### trainer.py: AlphaAtomsTrainer

```python
trainer = AlphaAtomsTrainer(network, config, tasks, initial_positions)
games = trainer.run_selfplay()   # returns list[Game], saves to replay buffer
result = trainer.fit()           # returns {'loss': float} or None (FakeNet/buffer too small)
trainer.save_checkpoint(path)    # saves via Fabric (model + optimizer state)
```

Fabric is set up once on first `fit()` call and reused. The trainer owns the Fabric, wrapped model, and optimizer as instance state.

### experiment.py

```python
run_id, run_dir = create_run_dir(config)          # creates outputs/<run_id>/, saves config.json
cost = compute_solution_cost(game)                 # replays game, sums reconfig + gate groups
solution = save_solution(game, path)               # writes atom-viz compatible JSON
metrics = selfplay_metrics(games, num_tasks)        # {best_cost, avg_cost, completion_rate, best_game}
append_to_registry(output_dir, run_id, config, m)  # appends to run_registry.jsonl
```

### game.py: Game

```python
game = Game(config, tasks, initial_positions)  # takes full config, accesses config.env/mcts/network
game.apply(action)                             # steps env, records reward/history
obs = game.make_observation(state_index)       # replays to reconstruct observation at step i
target = game.make_target(i, td_steps, to_play) # TD return + MCTS policy target
```

## Key Hyperparameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `mcts.num_simulations` | 50 | MCTS simulations per decision |
| `mcts.discount` | 1.0 | MCTS discount factor |
| `mcts.pb_c_base/init` | 19652 / 1.25 | UCB exploration constants |
| `training.num_selfplay` | 20 | Games per epoch |
| `training.buffer_size` | 1000 | Replay buffer capacity |
| `training.batch_size` | 128 | Training batch size |
| `training.training_steps` | 200 | Gradient steps per epoch |
| `training.td_steps` | 5 | TD bootstrap horizon |
| `training.lr` | 2e-4 | AdamW learning rate |
| `env.budget` | 24 | Max actions per episode |
| `network.num_bins` | 51 | Value distribution bins |
| `network.ema_decay` | 0.995 | Target network EMA |
| `network.v_hsize` | 64 | Value network hidden size |
| `network.p_hsize` | 32 | Policy network hidden size |
| `network.mlp_depth` | 2 | MLPMixer depth |
| `experiment.checkpoint_every_n_epochs` | 10 | Periodic checkpoint interval |
| `experiment.early_stopping_patience` | 10 | Epochs without improvement before stopping |

## Maps

Defined in `config.py` as `MAPS` list. Each map specifies:
- `board_dim`: (rows, cols)
- `num_qubits`: number of atoms
- `atom_map`: flat indices for initial atom placement
- `tasks`: list of gate layers, each a list of `[q1, q2]` pairs

Available maps: `0` = 2x6/9q, `1` = 4x4/8q, `2` = 5x5/12q. Selected via `--config.map_num=N`.

Derived values (set by `set_derived_config`): `env.board_height/width/num_qubits`, `network.num_tasks/num_qubits/board_size/num_actions`.

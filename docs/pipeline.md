# Training Pipeline Overview

## Problem

Given a grid of neutral atoms and a sequence of quantum gate layers (each layer = list of qubit pairs that must interact), find the optimal sequence of atom placements that minimizes the total number of parallel execution steps needed to complete all gate layers. Two atom moves can execute in parallel only if their paths don't cross rows or columns (AOD constraint).

## Layer-Level MDP

The environment operates as a **per-qubit sequential placement** model:

```
For each gate layer t (t = 0..num_tasks-1):
    relevant_atoms = sorted({q for pair in tasks[t] for q in pair})
    For each atom q in relevant_atoms (fixed order):
        Agent chooses action = flat_dest (0..board_size-1)
        → Place qubit q at cell flat_dest
        → Legal actions: empty cells + current cell (no-op)
    Layer t auto-executes (tasks_done++)
Episode ends when all layers are done.
```

| Property | Value |
|----------|-------|
| Action space | `board_size` (e.g., 12 for 2x6, 16 for 4x4) |
| Branching factor | ~9-15 (empty cells + current cell) |
| Episode length | `sum(k_t)` — deterministic |
| Completion rate | 100% by construction |

## Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  main.py                                                    │
│                                                             │
│  1. Load config (ml_collections ConfigDict + absl flags)    │
│  2. Derive map-specific values (board dims, action space)   │
│  3. Create run directory (outputs/<run_id>/)                │
│  4. Create Network + Trainer                                │
│  5. For each epoch: self-play → metrics → train → checkpoint│
│  6. Early stopping if best_cost plateaus                    │
│  7. Save best solution + final checkpoint + registry entry  │
└─────────────────────────────────────────────────────────────┘
```

### Epoch loop (trainer.py)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────────┐   │
│  │  Self-play   │───▶│  Replay Buffer   │───▶│  Gradient Training     │   │
│  │  (MCTS)      │    │  (TensorDict)    │    │  (Lightning Fabric)    │   │
│  └─────────────┘    └──────────────────┘    └────────────────────────┘   │
│                                                                          │
│  × num_selfplay       LazyTensorStorage       × training_steps           │
│    games/epoch         (capacity: 1000)          per epoch                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Self-play phase

For each game:

1. **Initialize** environment with atom positions and gate tasks
2. **MCTS search** at each step (one per qubit placement):
   - **Select**: Traverse tree using UCB scores (prior + value)
   - **Expand**: Clone environment, step action, query network for value + policy
   - **Backpropagate**: Update visit counts and value estimates (single-player: always add)
   - Repeat for `num_simulations` iterations
3. **Action selection**: Sample from visit count distribution (softmax with temperature decay)
4. **Step environment**: Place current qubit at chosen cell; if last qubit in layer, auto-execute
5. **Store statistics**: Visit count distribution (policy target) and root value
6. **Repeat** until all layers done

```
Environment state ──▶ make_features() ──▶ Network.inference()
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                     ▼
                              Policy logits          Value estimate
                              (board_size dim)       (correctness + latency)
```

### Feature construction (network.py: make_features)

```
Input: board_onehot (board_size × num_qubits+1), current_qubit

Slot 0: board state (one-hot qubit positions)
Slot 1..N: board state + gate pair indicators for each task layer
           (qubits involved in gates get +1.0 marker)

Output: (num_tasks+1, board_size, num_qubits) tensor

Qubit identity is injected after mixer1 in both ValueNetwork and PolicyNetwork
via deterministic sinusoidal embeddings (nqubits, dim), analogous to positional
encodings in transformers.
```

### Network architecture (network.py)

```
Features ──▶ NeutralAtomsMLP2
                  │
          ┌───────┴────────┐
          ▼                ▼
    ValueNetwork      PolicyNetwork
          │                │
    MLPMixer ×2       MLPMixer ×2
          │                │
    ┌─────┴─────┐         ▼
    ▼           ▼     W_pi ──▶ per-qubit logits (nqubits × board_size)
correctness  latency       │
 logits      logits    select current_qubit slice
    │           │           │
    └─────┬─────┘      log_softmax ──▶ policy_logits (board_size)
          │
   categorical
   distributions
   over value bins
```

- **Value head**: Outputs categorical distributions over `num_bins` bins (not scalar). Converted to scalar via expectation (`logits2values`).
- **Policy head**: `(batch, nqubits, board_size)` logits. At inference, current qubit's slice is selected → `(batch, board_size)`. During training, batched per-sample qubit indices are gathered.
- **Target network**: EMA shadow copy of the real network, used for bootstrap value estimation during training.

### Training phase (trainer.py: fit)

```
Replay Buffer ──sample batch──▶ Network.forward(batch)
                                      │
                               ┌──────┴──────┐
                               ▼              ▼
                         predictions    bootstrap predictions
                         (online net)   (EMA target net)
                               │              │
                               └──────┬───────┘
                                      ▼
                              Loss = policy_CE
                                   + correctness_value_CE
                                   + latency_value_CE
                                      │
                                      ▼
                              AdamW optimizer step
                              + gradient clipping
                              + EMA parameter update
```

Training targets per step `i`:
- **Policy target**: MCTS visit count distribution at step `i` (board_size-dim vector)
- **Correctness value target**: TD(n) return = `Σ(γ^k × r_{i+k})` + bootstrap from target network
- **Latency value target**: `-cost` if episode completed all tasks, else 0

### Reward signal (rewards.py)

```
reward = reward_scale * (prev_current_layer_cost - curr_current_layer_cost)
```

Where `current_layer_cost` = parallel execution cost of the **current gate layer only** given current atom positions (computed before auto-execute). Previous versions used the total remaining cost across all layers, but that telescoped to a constant cumulative reward regardless of actions, making it impossible for MCTS to distinguish good from bad trajectories. Using current-layer-only cost breaks the telescoping sum and provides a meaningful dense signal.

### Action space (env.py)

```
Action = flat_dest (0..board_size-1)
  = which cell to place the current qubit in
  Current cell = no-op (stay in place)
  Other empty cells = move qubit there
```

Legal actions: current cell + all empty cells. Occupied cells (by other atoms) are excluded.

### Config structure (config.py)

```
config
├── map_num          # Which map to use (0, 1, 2)
├── use_fake         # Use FakeNet for testing
├── env
│   ├── reward_scale
│   └── board_height, board_width, num_qubits  (derived from map)
├── mcts
│   ├── num_simulations, discount, max_moves
│   ├── pb_c_base, pb_c_init                   (UCB constants)
│   ├── root_dirichlet_alpha, root_exploration_fraction
│   ├── temperature_init, temperature_final, temperature_decay_steps
│   └── known_bounds.{min, max}
├── training
│   ├── epochs, num_selfplay, training_steps
│   ├── batch_size, lr, grad_norm_clip
│   ├── buffer_size, td_steps
│   ├── policy_target_temperature
│   ├── log_interval
│   └── accelerator, devices, seed
├── experiment
│   ├── output_dir                             (default: ./outputs)
│   ├── checkpoint_every_n_epochs              (default: 10)
│   └── early_stopping_patience                (default: 10)
└── network
    ├── v_hsize, p_hsize, mlp_depth
    ├── ema_decay, num_bins, value_min, value_max
    ├── correctness_weight, latency_weight
    └── num_tasks, num_qubits, board_size, num_actions  (derived from map)
```

### Experiment tracking (experiment.py)

```
outputs/
├── run_registry.jsonl              # one JSON line per completed run
└── <run_id>/                       # 8-char hex hash
    ├── config.json                 # frozen ConfigDict
    ├── metrics.jsonl               # per-epoch metrics
    ├── checkpoints/
    │   ├── epoch_010.ckpt          # periodic (every checkpoint_every_n_epochs)
    │   └── final.ckpt              # always saved at end
    └── solutions/
        ├── best.json               # atom-viz compatible (board + circuit + plan)
        └── best_trace.json         # per-step trace (action type, qubit, cost delta)
```

**Total cost** = reconfig parallel groups + gate execution groups (2x per layer for AOD enter/exit).

**Early stopping**: tracks `best_cost` across completed games per epoch. After `early_stopping_patience` consecutive epochs without improvement, training halts.

### Cost computation detail

Solution cost is computed by replaying the game and tracking actual moves per layer:

```
For each layer:
    reconfig_cost = count_groups(moves made during this layer)
    gate_cost = 2 × count_groups(gate interactions from final positions)
    total += reconfig_cost + gate_cost
```

The dense reward uses a per-layer heuristic: at each step, `_cached_cost` = parallel execution cost of the current gate layer given current atom positions (computed before auto-execute). Reward = prev_current_layer_cost - curr_current_layer_cost. This breaks the telescoping sum that plagued the old all-layers cost, allowing MCTS to distinguish good from bad action sequences.

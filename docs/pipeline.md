# Training Pipeline Overview

## Problem

Given a grid of neutral atoms and a sequence of quantum gate layers (each layer = list of qubit pairs that must interact), find the optimal sequence of atom moves that minimizes the total number of parallel execution steps needed to complete all gate layers. Two atom moves can execute in parallel only if their paths don't cross rows or columns (AOD constraint).

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
2. **MCTS search** at each step:
   - **Select**: Traverse tree using UCB scores (prior + value)
   - **Expand**: Clone environment, step action, query network for value + policy
   - **Backpropagate**: Update visit counts and value estimates up the path
   - Repeat for `num_simulations` iterations
3. **Action selection**: Sample from visit count distribution (softmax with temperature)
4. **Step environment**: Execute chosen action (move atom or execute gate layer)
5. **Store statistics**: Visit count distribution (policy target) and root value
6. **Repeat** until done or budget exhausted

```
Environment state ──▶ make_features() ──▶ Network.inference()
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                     ▼
                              Policy logits          Value estimate
                              (action priors)        (correctness + latency)
```

### Feature construction (network.py: make_features)

```
Input: board_onehot (board_size × num_qubits)

Slot 0: raw board state (reference)
Slot 1..N: board state + gate pair indicators for each task layer
           (qubits involved in gates get +1.0 marker)

Output: (num_tasks+1, board_size, num_qubits) tensor
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
    ▼           ▼     W_pi ──▶ move logits (per qubit × per cell)
correctness  latency        │
 logits      logits    gate_action logit
    │           │           │
    └─────┬─────┘     concat(gate_action, move_logits)
          │                │
   categorical         log_softmax ──▶ policy_logits
   distributions
   over value bins
```

- **Value head**: Outputs categorical distributions over `num_bins` bins (not scalar). Converted to scalar via expectation (`logits2values`).
- **Policy head**: One logit for "execute gate" action + `num_qubits × board_size` logits for move actions.
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
- **Policy target**: MCTS visit count distribution at step `i`
- **Correctness value target**: TD(n) return = `Σ(γ^k × r_{i+k})` + bootstrap from target network
- **Latency value target**: `-cost` if episode completed all tasks, else 0

### Reward signal (rewards.py)

```
reward = (prev_cost - curr_cost) × reward_scale
       + (prev_entropy - curr_entropy) × entropy_weight
```

Where `cost` = total parallel execution steps needed for all remaining gates given current atom positions. This provides dense reward: every move that improves future parallelism is immediately rewarded.

### Action space (env.py)

```
Action 0:              Execute current gate layer
Action 1 + q*S + p:    Move qubit q to cell p (S = board_size)
```

Legal actions exclude moves to occupied cells and are empty when all tasks are done or budget is exhausted.

### Config structure (config.py)

```
config
├── map_num          # Which map to use (0, 1, 2)
├── use_fake         # Use FakeNet for testing
├── env
│   ├── budget, entropy_weight, reward_scale
│   └── board_height, board_width, num_qubits  (derived from map)
├── mcts
│   ├── num_simulations, discount, max_moves
│   ├── pb_c_base, pb_c_init                   (UCB constants)
│   ├── root_dirichlet_alpha, root_exploration_fraction
│   └── known_bounds.{min, max}
├── training
│   ├── epochs, num_selfplay, training_steps
│   ├── batch_size, lr, grad_norm_clip
│   ├── buffer_size, td_steps
│   ├── log_interval
│   └── accelerator, devices, seed
├── experiment
│   ├── output_dir                             (default: ./outputs)
│   ├── checkpoint_every_n_epochs              (default: 10)
│   └── early_stopping_patience                (default: 10)
└── network
    ├── v_hsize, p_hsize, mlp_depth
    ├── ema_decay, num_bins, value_min, value_max
    └── num_tasks, num_qubits, board_size, num_actions  (derived from map)
```

### Experiment tracking (experiment.py)

```
outputs/
├── run_registry.jsonl              # one JSON line per completed run
└── <run_id>/                       # 8-char hex hash
    ├── config.json                 # frozen ConfigDict
    ├── checkpoints/
    │   ├── epoch_010.ckpt          # periodic (every checkpoint_every_n_epochs)
    │   └── final.ckpt              # always saved at end
    └── solutions/
        └── best.json               # atom-viz compatible (board + circuit + plan)
```

**Total cost** = reconfig parallel groups + gate execution groups (2x per layer for AOD enter/exit).

**Early stopping**: tracks `best_cost` across completed games per epoch. After `early_stopping_patience` consecutive epochs without improvement (only counting epochs with completions), training halts.

**Solution format** (atom-viz compatible):
- `board`: rows, cols, initialAtoms (dict of atom_id → {row, col})
- `circuit`: list of gate layers (list of qubit pairs)
- `plan`: list of parallel move groups, each group is list of {atom, from, to}

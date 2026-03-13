# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Onboarding (read in order)

1. **This file** — setup, commands, coding style, architecture overview
2. **`docs/pipeline.md`** — training pipeline with diagrams (feature construction, network architecture, training targets)
3. **`docs/architecture.md`** — detailed file structure, key interfaces, hyperparameter table, device lifecycle
4. **`experiments/README.md`** → individual round docs — what's been tested, results, known bottlenecks, next steps

### Current status

**Layer-level MDP** (Phase 0B complete). The "execute immediately" attractor from the old MDP is eliminated — there is no GATE_ACTION. The agent places atoms one at a time per layer, and layers auto-execute when all relevant atoms are placed. Episodes are deterministic length (`sum(k_t)` where `k_t` = relevant atoms per layer). Action space = `board_size` (~12-25) instead of the old `1 + Q * board_size` (~129). 100% completion rate by construction.

Next: **Round 04 experiments** (validate new MDP) and **Phase 1** (Iterative Refinement Model). See `docs/iterative_refinement_reference.md` and `roadmap.md`.

## Setup

Uses a uv-managed virtual environment at `../grid_mcts2/.venv`.

```bash
# Install dependencies
uv pip install torch lightning torchrl tensordict einops numpy ml_collections absl-py

# Activate for interactive use
source ../grid_mcts2/.venv/bin/activate
```

## Commands

```bash
# Run training (full)
../grid_mcts2/.venv/bin/python main.py --config.training.epochs=50 --config.training.num_selfplay=20 --config.training.batch_size=128

# Quick smoke test with FakeNet (no GPU, no real network)
../grid_mcts2/.venv/bin/python main.py --config.use_fake=True --config.training.epochs=2

# Small real-network test
../grid_mcts2/.venv/bin/python main.py --config.training.epochs=1 --config.training.num_selfplay=2 --config.training.batch_size=32 --config.training.training_steps=10

# Run tests (pytest on the moves/parallel-grouping + layer-level MDP tests)
../grid_mcts2/.venv/bin/python -m pytest neutral_atoms/test_env.py -v

# Select map (0=2x6/9q, 1=4x4/8q, 2=5x5/12q)
../grid_mcts2/.venv/bin/python main.py --config.map_num=1
```

Config uses `ml_collections.ConfigDict` with `absl` flags. Override any config value with `--config.<path>=<value>` (dot notation for nested fields). See `neutral_atoms/config.py:get_config()` for all available fields.

## Coding Style
- Compact code with minimal line breaks, no docstrings unless necessary (function name and variable names should be self-explanatory)
- Fail fast: avoid defensive try-except and if-else for edge cases unless explicitly needed. Do not assume any default values for missing attributes from configs.
- Prioritize clarity, efficiency, and soundness over over-engineering
- Helper functions should follow the single responsibility principle (least astonishment) with no side effects that allows for clean logic (does only what the function name suggests).
- Main function should be broken down with helper functions such that: 1) there is no duplicate code when possible, 2) with help of submodules and variables having self-explanatory naming conventions, basically resembles clean pseudocode and describes the high-level ideas it is trying to achieve
- Understand config parameters before testing, and always make sure to test with smaller model parameters, training time, etc. first for sanity checks

## Architecture

This is an AlphaZero-style MCTS training system for neutral atom quantum computing. The goal is to learn optimal atom reconfiguration sequences that minimize parallel execution steps for quantum gate layers.

### Layer-Level MDP

The environment uses a **per-qubit sequential placement** model:
- For each gate layer, `relevant_atoms = sorted({q for pair in tasks[layer] for q in pair})` gives k_t qubits
- The agent places them **one at a time in fixed order**
- **Action space = `board_size`** (which cell to place the current qubit in)
- Legal actions: empty cells + current cell (no-op = stay in place)
- After k_t placements, the layer auto-executes and `tasks_done` increments
- Episode length = `sum(k_t)` — deterministic, no budget needed
- **Branching factor: ~9-15** (empty cells) vs old 129 (all qubits x all cells + gate)

### Training loop (`main.py` → `trainer.py`)

Each epoch: **self-play** (MCTS generates games) → **compute metrics** → **train** (gradient steps on sampled batches) → **checkpoint/early-stop**.

- `run_selfplay()` returns a list of `Game` objects (also saves to replay buffer internally). Supports parallel self-play via `config.training.num_parallel_games` (uses `torch.multiprocessing.Pool`; each worker creates a fresh Network from state_dict).
- `fit()` returns `{'loss': float}` or `None` (FakeNet / buffer too small).
- `save_checkpoint(path)` saves model + optimizer via Fabric.
- `trainer.py` uses **Lightning Fabric** for device management (CPU/GPU) and **torchrl's `TensorDictReplayBuffer`** for experience storage.
- Training data is stored as `TensorDict` with nested keys: `('obs', 'features')`, `('obs', 'current_qubit')`, `('bootstrap_obs', 'features')`, `('target', 'correctness_values')`, etc.

### Device lifecycle

1. Network starts on CPU. First `fit()` call sets up Fabric once via `_setup_fabric()` and moves model to GPU.
2. Fabric accelerator defaults to `'auto'` — picks GPU when available, falls back to CPU.
3. Fabric, wrapped model, and optimizer persist as trainer state across epochs.
4. During self-play, `Network.inference()` moves CPU observations to the model's device automatically.
5. EMA shadow parameters synced to device via `EMA.to(device)` after `fabric.setup()`.
6. Environment always runs on CPU — this is correct and intentional.

### Network (`network.py`)

- `Network` wraps `NeutralAtomsMLP2` (real) or `FakeNet` (uniform random for testing).
- `NeutralAtomsMLP2` = `ValueNetwork` + `PolicyNetwork`, both using MLPMixer blocks.
- Input features: `(batch, num_tasks+1, board_size, num_qubits)` — slot 0 is board state, slots 1..N encode gate pair indicators. Qubit identity is injected after mixer1 in both ValueNetwork and PolicyNetwork via deterministic sinusoidal embeddings `(nqubits, dim)`, analogous to positional encodings in transformers.
- `PolicyNetwork` outputs `(batch, num_qubits, board_size)` — per-qubit policy logits. At inference, the current qubit's slice is selected. During training, batched `current_qubit` indices are used for gather.
- Value head outputs categorical distributions over bins (not scalar values). `logits2values()` converts via expectation.
- Target network uses EMA (`t_nnet`) with `average_parameters()` context manager for bootstrap value computation.

### Environment (`env.py`, `board.py`, `moves.py`, `rewards.py`)

- Actions: cell index (0..board_size-1) — where to place the current qubit
- Reward is dense: every step compares current-layer cost before vs after the action (computed before auto-execute). Uses current layer only (not all remaining layers) to break the telescoping sum that made cumulative reward constant.
- `compute_reward(prev_cost, curr_cost, reward_scale) -> float`
- Parallel grouping uses the AOD constraint: two atom moves can execute simultaneously only if they don't cross in rows or columns. Implemented as vectorized pairwise compatibility check + greedy graph coloring.

### MCTS (`mcts.py`)

Standard AlphaZero MCTS with environment cloning for simulation. Uses UCB selection, Dirichlet noise at root, linear temperature decay (`temperature_init` → `temperature_final` over `temperature_decay_steps`). Each simulation clones the environment and steps through it — no learned dynamics model. Single-player: `_backpropagate` always adds value (no negation).

## Config system (`neutral_atoms/config.py`)

- `get_config()` returns a `ml_collections.ConfigDict` with five sub-configs: `env`, `mcts`, `training`, `network`, `experiment`.
- `set_derived_config(config)` computes map-dependent values (board dims, num_qubits, action space size) and writes them into `config.env` and `config.network`.
- `config.network.num_actions = board_size` (action space is per-qubit cell selection).
- Map definitions (`MAPS`) and `atom_map_to_positions()` live in `config.py`.
- Top-level flags: `config.map_num`, `config.use_fake`.
- Temperature config: `mcts.temperature_init`, `mcts.temperature_final`, `mcts.temperature_decay_steps`.

## Experiment tracking (`neutral_atoms/experiment.py`)

Each run creates `outputs/<run_id>/` with:
- `config.json` — frozen ConfigDict snapshot
- `checkpoints/` — periodic + final model checkpoints (Lightning Fabric)
- `solutions/best.json` — atom-viz compatible JSON (board, circuit, plan)
- `solutions/best_trace.json` — per-step trace with action type (move/noop), qubit/src/dst, cost before/after, reward

`outputs/run_registry.jsonl` tracks all completed runs with config + metrics.

Early stopping: training halts if `best_cost` doesn't improve for `experiment.early_stopping_patience` epochs (only counts epochs with completed games).

Total cost metric = reconfig parallel groups + gate execution groups (2x per layer for enter/exit).

## Experiments

Experiment logs live in `experiments/`. Each round is a standalone markdown with hypotheses, exact reproducible commands, results, and analysis. See `experiments/README.md` for the index.

```bash
# Custom single run
../grid_mcts2/.venv/bin/python main.py --config.mcts.num_simulations=25 --config.training.epochs=15

# Check results
cat outputs/run_registry.jsonl
cat outputs/<run_id>/metrics.jsonl
```

Outputs go to `outputs/<run_id>/`. Each run saves `config.json`, `metrics.jsonl`, checkpoints, and best solution JSON.

## Map definitions

Maps are defined in `neutral_atoms/config.py` as dicts with `board_dim`, `num_qubits`, `atom_map`, and `tasks`. The `tasks` field is a list of gate layers, each layer being a list of qubit pairs.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Uses a uv-managed virtual environment. Always use `.venv/bin/python` (not system python).

```bash
# Install dependencies
uv pip install torch lightning torchrl tensordict einops numpy ml_collections absl-py

# Activate for interactive use
source .venv/bin/activate
```

## Commands

```bash
# Run training (full)
.venv/bin/python main.py --config.training.epochs=50 --config.training.num_selfplay=20 --config.training.batch_size=128

# Quick smoke test with FakeNet (no GPU, no real network)
.venv/bin/python main.py --config.use_fake=True --config.training.epochs=2

# Small real-network test
.venv/bin/python main.py --config.training.epochs=1 --config.training.num_selfplay=2 --config.training.batch_size=32 --config.training.training_steps=10

# Run tests (pytest on the moves/parallel-grouping tests)
.venv/bin/python -m pytest neutral_atoms/test_env.py -v

# Select map (0=2x6/9q, 1=4x4/8q, 2=5x5/12q)
.venv/bin/python main.py --config.map_num=1
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

### Training loop (`main.py` → `trainer.py`)

Each epoch: **self-play** (MCTS generates games) → **compute metrics** → **train** (gradient steps on sampled batches) → **checkpoint/early-stop**.

- `run_selfplay()` returns a list of `Game` objects (also saves to replay buffer internally).
- `fit()` returns `{'loss': float}` or `None` (FakeNet / buffer too small).
- `save_checkpoint(path)` saves model + optimizer via Fabric.
- `trainer.py` uses **Lightning Fabric** for device management (CPU/GPU) and **torchrl's `TensorDictReplayBuffer`** for experience storage.
- Training data is stored as `TensorDict` with nested keys: `('obs', 'features')`, `('bootstrap_obs', 'features')`, `('target', 'correctness_values')`, etc.

### Device lifecycle

1. Network starts on CPU. First `fit()` call sets up Fabric once via `_setup_fabric()` and moves model to GPU.
2. Fabric, wrapped model, and optimizer persist as trainer state across epochs.
3. During self-play, `Network.inference()` moves CPU observations to the model's device automatically.
4. EMA shadow parameters synced to device via `EMA.to(device)` after `fabric.setup()`.
5. Environment always runs on CPU — this is correct and intentional.

### Network (`network.py`)

- `Network` wraps `NeutralAtomsMLP2` (real) or `FakeNet` (uniform random for testing).
- `NeutralAtomsMLP2` = `ValueNetwork` + `PolicyNetwork`, both using MLPMixer blocks.
- Input features: `(batch, num_tasks+1, board_size, num_qubits)` — slot 0 is raw board, slots 1..N encode gate pair indicators.
- Value head outputs categorical distributions over bins (not scalar values). `logits2values()` converts via expectation.
- Target network uses EMA (`t_nnet`) with `average_parameters()` context manager for bootstrap value computation.

### Environment (`env.py`, `board.py`, `moves.py`, `rewards.py`)

- Actions: `0` = execute gate layer, `1+` = move qubit q to cell p (encoded as `1 + q * board_size + p`).
- Reward is dense: every step compares cost-before vs cost-after of executing all remaining gates, so moves that improve future parallelism are immediately rewarded.
- Parallel grouping uses the AOD constraint: two atom moves can execute simultaneously only if they don't cross in rows or columns. Implemented as vectorized pairwise compatibility check + greedy graph coloring.

### MCTS (`mcts.py`)

Standard AlphaZero MCTS with environment cloning for simulation. Uses UCB selection, Dirichlet noise at root, softmax temperature for action selection. Each simulation clones the environment and steps through it — no learned dynamics model.

## Config system (`neutral_atoms/config.py`)

- `get_config()` returns a `ml_collections.ConfigDict` with five sub-configs: `env`, `mcts`, `training`, `network`, `experiment`.
- `set_derived_config(config)` computes map-dependent values (board dims, num_qubits, action space size) and writes them into `config.env` and `config.network`.
- Map definitions (`MAPS`) and `atom_map_to_positions()` live in `config.py`.
- Top-level flags: `config.map_num`, `config.use_fake`.

## Experiment tracking (`neutral_atoms/experiment.py`)

Each run creates `outputs/<run_id>/` with:
- `config.json` — frozen ConfigDict snapshot
- `checkpoints/` — periodic + final model checkpoints (Lightning Fabric)
- `solutions/best.json` — atom-viz compatible JSON (board, circuit, plan)

`outputs/run_registry.jsonl` tracks all completed runs with config + metrics.

Early stopping: training halts if `best_cost` doesn't improve for `experiment.early_stopping_patience` epochs (only counts epochs with completed games).

Total cost metric = reconfig parallel groups + gate execution groups (2x per layer for enter/exit).

## Map definitions

Maps are defined in `neutral_atoms/config.py` as dicts with `board_dim`, `num_qubits`, `atom_map`, and `tasks`. The `tasks` field is a list of gate layers, each layer being a list of qubit pairs.

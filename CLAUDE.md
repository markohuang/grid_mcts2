# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Uses a uv-managed virtual environment. Always use `.venv/bin/python` (not system python).

```bash
# Install dependencies
uv pip install torch lightning torchrl tensordict einops numpy

# Activate for interactive use
source .venv/bin/activate
```

## Commands

```bash
# Run training (full)
.venv/bin/python main.py --epochs 50 --num_selfplay 20 --batch_size 128

# Quick smoke test with FakeNet (no GPU, no real network)
.venv/bin/python main.py --use_fake --epochs 2

# Small real-network test
.venv/bin/python main.py --epochs 2 --num_selfplay 2 --batch_size 32 --training_steps 10

# Run tests (pytest on the moves/parallel-grouping tests)
.venv/bin/python -m pytest neutral_atoms/test_env.py -v

# Select map (0=2x6/9q, 1=4x4/8q, 2=5x5/12q)
.venv/bin/python main.py --map_num 1
```

Key CLI args: `--num_sims`, `--num_selfplay`, `--training_steps`, `--epochs`, `--batch_size`, `--lr`, `--map_num`, `--use_fake`, `--budget`.

## Architecture

This is an AlphaZero-style MCTS training system for neutral atom quantum computing. The goal is to learn optimal atom reconfiguration sequences that minimize parallel execution steps for quantum gate layers.

### Training loop (`main.py` → `trainer.py`)

Each epoch: **self-play** (MCTS generates games using current network) → **save to replay buffer** → **train** (gradient steps on sampled batches).

- `trainer.py` uses **Lightning Fabric** for device management (CPU/GPU) and **torchrl's `TensorDictReplayBuffer`** for experience storage.
- Training data is stored as `TensorDict` with nested keys: `('obs', 'features')`, `('bootstrap_obs', 'features')`, `('target', 'correctness_values')`, etc. The `Network.forward(batch)` accesses these as `batch['obs']['features']`.

### Device lifecycle

1. Network starts on CPU. First `fit()` call moves it to GPU via `fabric.setup()`.
2. During subsequent self-play, the network stays on GPU. `Network.inference()` moves CPU observations to the model's device automatically.
3. EMA shadow parameters (target network) are synced to device via `EMA.to(device)` after `fabric.setup()`.
4. Environment always runs on CPU — this is correct and intentional.

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

## Map definitions

Maps are defined in `main.py` as dicts with `board_dim`, `num_qubits`, `atom_map`, and `tasks`. The `tasks` field is a list of gate layers, each layer being a list of qubit pairs.

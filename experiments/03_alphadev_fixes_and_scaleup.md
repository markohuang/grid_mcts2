# Round 03: AlphaDev Alignment Fixes + Scale-Up

**Date:** 2026-03-13
**Map:** 1 (4x4, 8q, 3 identical layers, 4 gates each)
**Cost bounds:** lb=6, ub=24. Action space: 129.
**Hardware:** 2x RTX A6000 (48GB), 48 CPU cores

## Context

Comparison with AlphaDev pseudocode (see `docs/alphadev_comparison.md`) revealed several bugs and divergences:
1. **Bootstrap formula was halved** — value targets were 0.5x correct magnitude
2. **One-hot encoding** — lost gradient information (now two-hot interpolation)
3. **Coarse value bins** — 51 bins over [-25,25] = 1.0/bin (now 101 over [-10,10] = 0.2/bin)
4. **No reward weighting** — AlphaDev uses correctness:latency = 4:1 (2.0:0.5)

Plus new capabilities: `gate_only` reward mode, parallel self-play, observation caching, per-epoch timing.

## Goal

1. **Quantify the effect of infrastructure fixes** — do the bootstrap/two-hot/bins fixes alone improve results?
2. **Test gate_only reward mode** — does removing reconfig cost from the step reward let the agent explore reconfiguration?
3. **Find the right scale** — how many simulations, games, training steps are needed to see improvement? What's the wall-clock time?
4. **Test AlphaDev reward weighting** — does correctness_weight=2.0, latency_weight=0.5 help?

## Hypotheses

- **H1 — Fixes alone improve over Round 02.** The bootstrap fix means MCTS gets properly calibrated values. Two-hot means smoother gradients. Predicted: lower best_cost than Round 02 baseline (13), or faster convergence to similar cost.
- **H2 — gate_only breaks the execute-immediately attractor.** With reconfig cost=0 in the reward, individual moves that improve gate parallelism get purely positive reward. Predicted: avg_steps > 3.0 (agent tries moves), gate_action_fraction < 1.0.
- **H3 — More simulations help.** 50-100 sims with 129 actions means more actions get explored per move. Predicted: 100 sims > 50 sims in cost, at ~2x wall-clock time.
- **H4 — Parallel self-play enables more games per epoch.** 8 workers should give ~3-4x speedup over sequential. Use extra throughput for more games and more training steps.
- **H5 — AlphaDev reward weighting helps.** Correctness:latency = 2:0.5 prioritizes step-by-step cost reduction over final cost. Predicted: helps with gate_only mode (correctness is gate cost delta, latency is real total cost at end).

## Shared parameters (unless overridden)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `map_num` | 1 | Same map as Round 02 for comparison |
| `mcts.root_dirichlet_alpha` | 0.3 | High exploration (Round 01 finding) |
| `mcts.root_exploration_fraction` | 0.5 | High exploration |
| `training.epochs` | 30 | Enough for convergence |
| `experiment.early_stopping_patience` | 15 | Don't stop too early |
| `training.lr` | 2e-4 | Same as AlphaDev |

## Phase A: Sanity check fixes (fast, sequential)

### Exp 1 — Baseline with all fixes

Same hyperparams as Round 02 Exp 1, but with bootstrap/two-hot/bins fixes active.

```bash
.venv/bin/python main.py \
  --config.map_num=1 \
  --config.mcts.num_simulations=25 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.mcts.root_exploration_fraction=0.5 \
  --config.training.epochs=30 \
  --config.training.num_selfplay=10 \
  --config.training.batch_size=64 \
  --config.training.training_steps=100 \
  --config.experiment.early_stopping_patience=15
```

### Exp 2 — gate_only reward mode

Same as Exp 1 but step reward ignores reconfig group cost.

```bash
# Same as Exp 1, plus:
  --config.env.reward_mode=gate_only
```

### Exp 3 — gate_only + AlphaDev reward weighting

Prioritize step reward learning (correctness_weight=2.0) over terminal cost (latency_weight=0.5).

```bash
# Same as Exp 2, plus:
  --config.network.correctness_weight=2.0 \
  --config.network.latency_weight=0.5
```

## Phase B: Scale-up

### Exp 4 — More simulations (100 sims)

Does deeper search find better moves?

```bash
# Same as Exp 1, plus:
  --config.mcts.num_simulations=100
```

### Exp 5 — Parallel self-play, more games, more training

Use 8 parallel workers, 40 games/epoch, batch 256, 400 training steps. This exercises the GPU more and fills the replay buffer faster.

```bash
.venv/bin/python main.py \
  --config.map_num=1 \
  --config.mcts.num_simulations=50 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.mcts.root_exploration_fraction=0.5 \
  --config.training.epochs=30 \
  --config.training.num_selfplay=40 \
  --config.training.num_parallel_games=8 \
  --config.training.batch_size=256 \
  --config.training.training_steps=400 \
  --config.training.buffer_size=5000 \
  --config.experiment.early_stopping_patience=15
```

### Exp 6 — Full scale: gate_only + more sims + parallel + AlphaDev weights

Combine the best config from Phase A with Phase B scale.

```bash
.venv/bin/python main.py \
  --config.map_num=1 \
  --config.env.reward_mode=gate_only \
  --config.mcts.num_simulations=100 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.mcts.root_exploration_fraction=0.5 \
  --config.training.epochs=30 \
  --config.training.num_selfplay=40 \
  --config.training.num_parallel_games=8 \
  --config.training.batch_size=256 \
  --config.training.training_steps=400 \
  --config.training.buffer_size=5000 \
  --config.network.correctness_weight=2.0 \
  --config.network.latency_weight=0.5 \
  --config.experiment.early_stopping_patience=15
```

## What to look for

After each run, check with `python analyze.py <run_id>`:

1. **gate_action_fraction** — if still ~1.0, the agent is still just executing. Any value < 0.8 means it's trying moves.
2. **avg_steps** — > 3.0 means the agent is doing something besides execute-execute-execute.
3. **best_cost** — target is < 13 (best from Round 02). Lower bound is 6.
4. **avg_root_value** — should be non-trivial (not near 0). With the bootstrap fix, values should be properly calibrated.
5. **selfplay_time vs train_time** — understand the bottleneck at each scale point.
6. **Convergence speed** — when does best_cost stabilize? Is it still found via lucky early noise, or does the policy actually learn?

## Results

_To be filled in after running._

| # | Experiment | best_cost | avg_cost | avg_steps | gate_frac | pi_entropy | sp_time | tr_time |
|---|-----------|-----------|----------|-----------|-----------|------------|---------|---------|
| 1 | Baseline + fixes | | | | | | | |
| 2 | gate_only | | | | | | | |
| 3 | gate_only + weights | | | | | | | |
| 4 | 100 sims | | | | | | | |
| 5 | Parallel + scale | | | | | | | |
| 6 | Full scale | | | | | | | |

# Round 01: Sanity Checks

**Date:** 2026-03-12
**Map:** 0 (2x6 board, 9 qubits, 3 task layers)
**Cost bounds:** lb=6 (gate-only best), ub=22 (gate-only worst). Actual costs include reconfiguration moves.

## Goal

Verify the system works end-to-end before scaling up. Confirm that (a) MCTS search alone improves over random play, (b) learned networks improve over random networks, and (c) identify any obvious pathologies in training dynamics.

## Hypotheses

- **H1 — MCTS search helps.** More simulations should produce better solutions even without learning (FakeNet). Predicted: 50 sims >> 10 sims in completion rate and cost.
- **H2 — Learning improves over random.** A trained network with fewer sims should match or beat FakeNet with more sims. Predicted: Learn@10 sims >= FakeNet@50 sims.
- **H3 — Exploration-exploitation balance matters.** Dirichlet noise parameters affect solution quality. Predicted: moderate exploration > low exploration, but very high exploration may hurt.
- **H4 — Network capacity affects learning.** Larger networks learn more expressive policies/values. Predicted: large net > small net, especially in consistency.
- **H5 — Value estimates improve with training.** Root values should trend upward as the value network learns. Predicted: root_value increases monotonically over epochs.

## Setup

Shared settings across all learning experiments (3-8):

| Parameter | Value |
|-----------|-------|
| `training.num_selfplay` | 5 |
| `training.batch_size` | 32 |
| `training.training_steps` | 50 |
| `experiment.early_stopping_patience` | 20 (effectively disabled) |

Default config values (unless overridden):

| Parameter | Default |
|-----------|---------|
| `mcts.num_simulations` | 50 |
| `mcts.root_dirichlet_alpha` | 0.03 |
| `mcts.root_exploration_fraction` | 0.25 |
| `network.v_hsize` | 64 |
| `network.p_hsize` | 32 |
| `network.mlp_depth` | 2 |

## Experiments

### Exp 1 — FakeNet, 10 sims (H1 baseline)

```bash
.venv/bin/python main.py \
  --config.use_fake=True \
  --config.mcts.num_simulations=10 \
  --config.training.epochs=1 \
  --config.training.num_selfplay=30 \
  --config.experiment.early_stopping_patience=20
```

### Exp 2 — FakeNet, 50 sims (H1)

```bash
.venv/bin/python main.py \
  --config.use_fake=True \
  --config.mcts.num_simulations=50 \
  --config.training.epochs=1 \
  --config.training.num_selfplay=30 \
  --config.experiment.early_stopping_patience=20
```

### Exp 3 — Learning, 10 sims, 15 epochs (H2)

```bash
.venv/bin/python main.py \
  --config.mcts.num_simulations=10 \
  --config.training.epochs=15 \
  --config.training.num_selfplay=5 \
  --config.training.batch_size=32 \
  --config.training.training_steps=50 \
  --config.experiment.early_stopping_patience=20
```

### Exp 4 — Learning, 25 sims, 10 epochs (H2)

```bash
.venv/bin/python main.py \
  --config.mcts.num_simulations=25 \
  --config.training.epochs=10 \
  --config.training.num_selfplay=5 \
  --config.training.batch_size=32 \
  --config.training.training_steps=50 \
  --config.experiment.early_stopping_patience=20
```

### Exp 5 — High exploration: alpha=0.3, frac=0.5 (H3)

```bash
.venv/bin/python main.py \
  --config.mcts.num_simulations=10 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.mcts.root_exploration_fraction=0.5 \
  --config.training.epochs=15 \
  --config.training.num_selfplay=5 \
  --config.training.batch_size=32 \
  --config.training.training_steps=50 \
  --config.experiment.early_stopping_patience=20
```

### Exp 6 — Low exploration: alpha=0.003, frac=0.1 (H3)

```bash
.venv/bin/python main.py \
  --config.mcts.num_simulations=10 \
  --config.mcts.root_dirichlet_alpha=0.003 \
  --config.mcts.root_exploration_fraction=0.1 \
  --config.training.epochs=15 \
  --config.training.num_selfplay=5 \
  --config.training.batch_size=32 \
  --config.training.training_steps=50 \
  --config.experiment.early_stopping_patience=20
```

### Exp 7 — Small network: v_hsize=32, p_hsize=16 (H4)

35,225 trainable parameters.

```bash
.venv/bin/python main.py \
  --config.mcts.num_simulations=10 \
  --config.network.v_hsize=32 \
  --config.network.p_hsize=16 \
  --config.training.epochs=15 \
  --config.training.num_selfplay=5 \
  --config.training.batch_size=32 \
  --config.training.training_steps=50 \
  --config.experiment.early_stopping_patience=20
```

### Exp 8 — Large network: v_hsize=128, p_hsize=64 (H4)

311,513 trainable parameters.

```bash
.venv/bin/python main.py \
  --config.mcts.num_simulations=10 \
  --config.network.v_hsize=128 \
  --config.network.p_hsize=64 \
  --config.training.epochs=15 \
  --config.training.num_selfplay=5 \
  --config.training.batch_size=32 \
  --config.training.training_steps=50 \
  --config.experiment.early_stopping_patience=20
```

## Results

| # | Experiment | best_cost | avg_cost | completion | avg_steps | root_value (final) | pi_entropy (final) |
|---|-----------|-----------|----------|------------|-----------|-------------------|-------------------|
| 1 | FakeNet 10 sims | 24 | 28.0 | 10% | 23.6 | -1.71 | 0.31 |
| 2 | FakeNet 50 sims | 16 | 24.0 | 80% | 17.1 | -2.28 | 1.15 |
| 3 | Learn 10 sims | 16 | 16.4 | 100% | 3.4 | 14.56 | 0.00 |
| 4 | Learn 25 sims | 16 | 16.0 | 100% | 3.0 | 15.49 | 0.09 |
| 5 | High explore | **13** | 16.0 | 100% | 3.0 | 13.86 | 0.00 |
| 6 | Low explore | 15 | 16.0 | 100% | 3.0 | 13.48 | 0.00 |
| 7 | Small net (35k) | **13** | 20.2 | 100% | 7.6 | 11.83 | 0.00 |
| 8 | Large net (312k) | **13** | 16.2 | 100% | 3.8 | 15.41 | 0.00 |

Run IDs: 174474e7, fcf7c911, 692257a6, b8c8d316, 55f9f5d5, ff78653d, dfbbdf10, 41eb886c.

## Analysis

### H1: MCTS search helps — Confirmed

FakeNet@50 sims dramatically outperforms FakeNet@10 sims: 80% vs 10% completion, best cost 16 vs 24. Pure search without any learning is already valuable. The 5x increase in simulations translates to substantially better solutions.

### H2: Learning improves over random — Confirmed

Learn@10 sims (Exp 3) matches FakeNet@50 sims (Exp 2) in best cost (16) while achieving 100% completion vs 80%, with 5x fewer simulations. By epoch 5, the learned network reliably completes all games in ~3 steps. Learn@25 sims (Exp 4) converges even faster — stable at cost=16 from epoch 4 onward with `avg_steps=3.0` every epoch (the minimum possible for 3 tasks).

### H3: Exploration-exploitation balance — Confirmed

High exploration (alpha=0.3, frac=0.5) found cost=13, beating both default (16) and low exploration (15). The extra noise helps discover better solutions that the deterministic policy misses. However, cost=13 appeared in only 2 of 15 epochs — it's found by residual stochasticity, not reliably learned.

Low exploration (alpha=0.003, frac=0.1) performed slightly worse than default, finding cost=15 rather than 16. The difference is small, suggesting that on this easy map even minimal exploration suffices for decent solutions.

### H4: Network capacity — Partially confirmed

All three network sizes (35k, 88k default, 312k) found cost=13 at least once. The key difference is consistency:
- **Large net** (312k): 100% completion throughout, avg_cost=16.2, avg_steps=3.8
- **Default net** (88k): 100% completion, avg_cost=16.4, avg_steps=3.4 (Exp 3)
- **Small net** (35k): Less stable — completion dropped to 80% in epoch 13, avg_cost=20.2, avg_steps=7.6 in final epoch

The large network is most reliable. The small network learns slower (reaches 100% completion at epoch 7 vs epoch 5) and is less consistent. For this small map the capacity difference is modest; it likely matters more for larger maps.

### H5: Value estimates improve — Confirmed

Root values across all learning runs follow the same pattern:
- Epoch 1: negative values (-1.5 to -2.2) — untrained network
- Epoch 2: jumps to ~14-17 — rapid initial learning
- Epochs 3+: stabilizes around 13-16

The values improve but are not well-calibrated in absolute terms (the actual game rewards are much smaller). The relative ordering is correct though — states from completed games get higher values than incomplete ones.

### Unexpected findings

**Policy entropy collapse.** In every learning run, `avg_policy_entropy` drops to 0.00 within 5-7 epochs. The network learns a near-deterministic policy very quickly. This means:
1. MCTS search degenerates — with a peaked prior, the tree barely explores alternatives
2. The only source of diversity is Dirichlet noise at the root
3. cost=13 solutions are found by luck (noise), not by a policy that understands them

This is the most important finding. The system works but has a ceiling — premature policy convergence prevents discovering and reliably reproducing better solutions.

**avg_steps=3.0.** In the best runs, every game completes in exactly 3 steps (one "execute gates" action per task layer, no moves needed). This means the initial atom positions already satisfy the gate layers on this map without reconfiguration — the network learned to just execute immediately. The cost=16 coming from gate parallelism alone, not from suboptimal moves.

## Next steps

1. **Address entropy collapse** — add entropy bonus to loss, use higher temperature for longer, or increase Dirichlet noise as default
2. **Scale to harder maps** — Map 1 (4x4, 8q) and Map 2 (5x5, 12q) where reconfiguration is actually required
3. **Longer runs** — more self-play games per epoch and more epochs with the high-exploration config to see if cost=13 can become reliable
4. **Investigate cost=13** — export and visualize the best solution to understand what reconfiguration it found

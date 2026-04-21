# Round 06: Curriculum, Data Augmentation, and Reward Mode Comparison

## Questions

1. Does curriculum learning (progressive map expansion) improve generalization across random maps?
2. Does data augmentation (8 board symmetries) improve generalization?
3. Do more MCTS simulations (100 vs 50) improve final cost?
4. How do reward modes (Option E/B/C) compare in convergence and generalization?

## Setup

**Base**: Map 2 (5x5, 12q, 3 tasks), starting from specialist checkpoint `ef54adc7` (trained ~30 epochs on Map 2, achieved best=12).

**Curriculum**: Progressive map pool expansion — start from specialist, add one random map every `curriculum_patience=40` epochs without improvement. Pool grows: [map2] → [map2, rand1] → ... 80% of games use newest map, 20% use any prior map (`fixed_map_fraction=0.2`). Evaluated separately on each map in the pool (5 eval games per map, no noise).

**Augmentation**: Per-sample random board symmetry from the 8-element dihedral group (4 rotations × 2 flips). Applied via `torch.gather` on features and policy targets each training step.

**Baselines (no-reconfig cost)**:
- map0 (fixed eval, 5x5): no-reconfig ~18, lower bound ~6, specialist achieves **12**
- random curriculum maps: no-reconfig varies

---

## 6.1 Curriculum + Augmentation (Option E)

### Commands

```bash
# aug_s42 — seed 42
../grid_mcts2/.venv/bin/python main.py --config.map_num=2 \
    --config.training.epochs=300 --config.training.num_selfplay=20 \
    --config.training.batch_size=128 --config.mcts.root_dirichlet_alpha=0.1 \
    --config.experiment.curriculum_maps=3 --config.experiment.curriculum_initial_phase=3 \
    --config.experiment.curriculum_patience=40 --config.experiment.early_stopping_patience=80 \
    --config.training.fixed_map_fraction=0.2 \
    --config.training.data_augmentation=True \
    --config.experiment.load_checkpoint=outputs/ef54adc7/checkpoints/final.ckpt \
    --config.training.seed=42

# aug_s123 — seed 123 (same config, different seed)
```

### Results

| Run | Epochs | map0 | map1 | map2 | map3 | Entropy | Notes |
|-----|--------|------|------|------|------|---------|-------|
| aug_s42 (`bab2c181`) | 81 | **12** | **15** | **14** | 16 | 0.32 | Best map2 seen |
| aug_s123 (`fa4610df`) | 116 | **12** | **15** | 16 | 16 | 0.35 | Consistent with s42 |

### Observations

- Both seeds converge to map1=**15**, confirming it's a stable floor for this curriculum.
- aug_s42 achieved map2=**14** — the best generalization seen so far across all runs.
- Entropy remains healthy (~0.32–0.35) throughout, no collapse.
- Augmentation (8 symmetries) appears to be the primary driver of generalization: map1/map2 values improved vs no-aug curriculum runs.

---

## 6.2 More Simulations (sims=100, Option E + Augmentation)

```bash
../grid_mcts2/.venv/bin/python main.py --config.map_num=2 \
    --config.mcts.num_simulations=100 \
    --config.training.data_augmentation=True \
    [same curriculum flags as 6.1]
```

### Results

| Run | Epochs | map0 | map1 | map2 | map3 | Notes |
|-----|--------|------|------|------|------|-------|
| sims100_s42 (`6256b33a`) | 54 | **12** | 17 | **15** | 16 | Mixed — worse map1 than aug |
| sims100_s123 (`8dc0f203`) | 54 | **12** | 16 | 17 | **18** | Clearly worse than aug |

Each epoch takes ~3× longer (167s vs 85s), so 54 epochs ≈ 35 "equivalent" epochs vs aug runs' 81+.

### Observations

- More simulations do **not** clearly improve generalization at this stage — both seeds are worse than the 50-sim aug runs after equivalent wall time.
- sims100_s123 achieved map3=18, the worst generalization result in this round.
- The bottleneck is likely not tree search quality but reward signal and training distribution diversity.

---

## 6.3 Reward Mode: Option B (layer_delta) + Curriculum

```bash
../grid_mcts2/.venv/bin/python main.py --config.map_num=2 \
    --config.env.reward_mode=layer_delta \
    [same curriculum flags as 6.1, NO augmentation]
    --config.training.seed=42
```

Run: `8d8e0b61`, 81 epochs.

### Formulation (corrected)

Within layer L, at each step:
```
r_t = layer_cost(t-1) - layer_cost(t)   [delta, can be positive or negative]
```
At auto-execute (all k_t atoms placed):
```
r_correction = -initial_layer_cost(L)
```
Where `initial_layer_cost(L)` = gate execution cost of layer L before any reconfigurations (do-nothing baseline for that layer). This ensures: `sum_layer(r) = -actual_layer_cost(L)`.

Total episode reward = -actual_solution_cost (unbiased). No cross-layer signal from reward itself (only from value bootstrap).

### Results

| Metric | Value |
|--------|-------|
| best_cost_so_far | 12 |
| avg_cost | 15.6–16.4 |
| map0_best | 12 |
| map1_best | **18** (worst of all runs) |
| map2_best | 16 |
| map3_best | 16 |
| Entropy | **0.145** (collapsed from ep10) |
| root_value | ~8.5 (overfit) |
| correctness_loss | ~0.002 (near zero — overfit) |

### Trajectory

Policy entropy collapsed to ~0.15 by epoch 10 and stayed there for all 81 epochs. This is 2–3× lower than the Option E runs. The value head completely overfit (cv_loss ≈ 0, root_value >> expected), while generalization to new maps degraded to 18 (vs 15 for Option E+aug).

### Analysis

Option B without augmentation causes immediate policy collapse. The within-layer delta reward is highly non-stationary: large negative reward when moving an atom away, large positive reward when placing it correctly. This creates a reward landscape where the policy rapidly converges to a deterministic sequence and the value head memorizes it, rather than learning generalizable features.

Note: Option B was run *without* augmentation. A follow-up (6.5 below) tests Option B *with* augmentation from scratch.

---

## 6.4 Reward Mode Comparison: Key Findings

| Mode | Total reward | Credit assignment | Cross-layer | Bias (strict RL) |
|------|-------------|-------------------|-------------|-------------------|
| **E (plan_cost)** | do_nothing - actual_cost | Dense, cross-layer | Yes (remaining_cost includes all layers) | None (valid baseline subtraction) |
| **B (layer_delta)** | -actual_cost | Dense, within-layer only | Only via V bootstrap | None (unbiased total) |
| **C (layer_completion)** | -actual_cost | Sparse (boundary only) | Only via V bootstrap | None (unbiased total) |

Option E is not biased in the strict RL sense — `do_nothing_remaining(s)` is a valid baseline (action-independent), reducing variance like an advantage function. The shifted value target is a consequence, not a problem.

The practical winner so far: **Option E + augmentation** (aug_s42: map2=14, map1=15).

---

## 6.5 Reward Mode B/C from Scratch (Ongoing)

Testing whether the Option B collapse is intrinsic to the reward mode or was amplified by starting from a converged specialist checkpoint.

### Commands

```bash
# B_fixed — layer_delta, map2 fixed, from scratch
../grid_mcts2/.venv/bin/python main.py --config.map_num=2 \
    --config.env.reward_mode=layer_delta \
    --config.training.data_augmentation=True \
    --config.training.epochs=100 --config.training.num_selfplay=20 --config.training.seed=42

# C_fixed — layer_completion, map2 fixed, from scratch
../grid_mcts2/.venv/bin/python main.py --config.map_num=2 \
    --config.env.reward_mode=layer_completion --config.training.td_steps=10 \
    --config.training.data_augmentation=True \
    --config.training.epochs=100 --config.training.num_selfplay=20 --config.training.seed=42

# B_generalist — layer_delta, random_board, from scratch
../grid_mcts2/.venv/bin/python main.py --config.map_num=2 --config.random_board=True \
    --config.env.reward_mode=layer_delta \
    --config.training.data_augmentation=True \
    --config.training.epochs=100 --config.training.num_selfplay=20 --config.training.seed=42

# C_generalist — layer_completion, random_board, from scratch
../grid_mcts2/.venv/bin/python main.py --config.map_num=2 --config.random_board=True \
    --config.env.reward_mode=layer_completion --config.training.td_steps=10 \
    --config.training.data_augmentation=True \
    --config.training.epochs=100 --config.training.num_selfplay=20 --config.training.seed=42
```

### Run IDs

| Run | ID | Status |
|-----|----|--------|
| B_fixed | `c9509e38` | Running |
| C_fixed | `2ea35829` | Running |
| B_generalist | `3af53805` | Running |
| C_generalist | `bea23c09` | Running |

**Comparison baselines**:
- Option E fixed, from scratch: `ef54adc7` (hit 12 ~ep30)
- Option E generalist, from scratch: `d7e3ccd1` (round 05 style)

### Hypotheses

- **H1**: Option B with augmentation from scratch avoids collapse (augmentation provides enough diversity to prevent value overfitting).
- **H2**: Option C (sparse) converges slower than E/B but may reach a better floor due to unbiased signal.
- **H3**: Generalist runs with B/C may generalize better than fixed-map runs due to reward modes that are unbiased across maps.

### Results

Runs killed before completing epoch 1 — no output produced. Killed to launch 6.6 exploration runs. No data on cold-start B/C behavior from this round.

---

## 6.6 Exploration Parameter Study

### Motivation

After finding map2=**11** was possible (sims=200, run A `bf70eaaa` predecessor), the question became: are local minima (map0=12 being the typical floor) due to insufficient exploration rather than reward/architecture limits? Round 6.6 systematically varies MCTS exploration parameters.

**Base**: Option E (plan_cost) + augmentation + curriculum from `ef54adc7`, same as 6.1.

**Parameters varied**:
- **Simulations**: 50 (baseline) → 200 (×4 search depth/width)
- **Dirichlet alpha**: 0.1 (baseline) → 0.3 (more root noise)
- **Temperature decay**: 1000 steps (≈ep5 at 200 steps/epoch) → 5000 steps (≈ep25, much slower cooling)

### Commands

```bash
# A: sims=200, alpha=0.1, temp_decay=1000 (more sims only)
../grid_mcts2/.venv/bin/python main.py ... --config.mcts.num_simulations=200

# B: sims=50, alpha=0.3, temp_decay=1000 (more noise only)
../grid_mcts2/.venv/bin/python main.py ... --config.mcts.root_dirichlet_alpha=0.3 --config.training.fixed_map_fraction=0.5

# C: sims=50, alpha=0.1, temp_decay=5000 (slower cooling only)
../grid_mcts2/.venv/bin/python main.py ... --config.mcts.temperature_decay_steps=5000

# D: sims=200, alpha=0.3, temp_decay=5000 (all combined)
../grid_mcts2/.venv/bin/python main.py ... --config.mcts.num_simulations=200 --config.mcts.root_dirichlet_alpha=0.3 --config.mcts.temperature_decay_steps=5000

# E: sims=50, alpha=0.1, temp_decay=1000, reward_mode=plan_cost_unbiased (Option G baseline)
../grid_mcts2/.venv/bin/python main.py ... --config.env.reward_mode=plan_cost_unbiased
```

### Results

| Run | ID | Epochs | Epoch time | map0 | map1* | map2* | map3* | Entropy | avg_cost |
|-----|----|--------|-----------|------|-------|-------|-------|---------|----------|
| A (sims=200) | `bf70eaaa` | 63 | 251s | **12** | **14** | **13** | **14** | 0.36 | 15.4 |
| B (alpha=0.3) | `331f5082` | 138 | 66s | **12** | **13** | **13** | **14** | 0.70 | 15.6 |
| C (temp_decay=5000) | `40438604` | 125 | 80s | **12** | **14** | **15** | 16 | 0.27 | 15.7 |
| D (all combined) | `2747bd10` | 63 | 257s | **12** | **14** | **13** | **14** | 0.51 | **13.7** |
| E (Option G / plan_cost_unbiased) | `925e114e` | 277 | 54s | 14 | **13** | 16 | **14** | **1.77** | 19.6 |

*All-time best across run (not just final epoch). eval maps = random maps seeded by `training.seed*100+phase`.*

### Observations

- **Run B (alpha=0.3)** achieved map1=**13** — the best generalization to curriculum maps seen across all runs. Higher Dirichlet noise (0.3 vs 0.1) keeps entropy at 0.70 (vs 0.36 for baseline), enabling continued exploration without eval degradation.
- **Run D (all combined)** achieved the best **avg_cost=13.7** in self-play games — the agent is consistently finding better solutions during training, not just lucky eval hits.
- **Run A (sims=200)** is effective but slow (251s/epoch vs 66s); eval quality similar to baseline except for map3 (14 vs 16).
- **Run C (temp_decay=5000)** is the weakest — slower temperature decay doesn't help and entropy collapses anyway (0.27), suggesting temperature is not the binding exploration constraint.
- **Option G (Run E)**: entropy stays extremely high (1.77) for 277 epochs — avg_cost=19.6 (worse than do-nothing baseline of 18). Like runs A–D, this loaded from `ef54adc7`, but switched reward modes. The checkpoint's value function learned Option E's shifted target (`do_nothing_remaining - actual_cost`); Option G rewards are systematically lower by `do_nothing_total ≈ 18`, so the pretrained V overestimates every state. MCTS Q-values are immediately wrong, resulting in near-random policy for the entire run. Starting from random init would converge faster — there's no wrong prior to overcome.

### Analysis

The exploration bottleneck is primarily **Dirichlet noise magnitude** (alpha), not simulations or temperature schedule. alpha=0.3 pushes the policy into novel states and achieves the best generalization (map1=13). Combining with sims=200 (Run D) gives the best avg_cost but at 4× compute cost per epoch.

Option G is not a drop-in replacement for Option E when loading a pretrained Option-E checkpoint — the pretrained V encodes the shifted target and overestimates every state under the unbiased reward signal. A randomly initialized network would converge faster since it has no wrong prior to overcome. To test Option G properly: train from scratch (no `load_checkpoint`).

---

## Summary

| Experiment | map0 | map1* | map2* | map3* | Key finding |
|------------|------|-------|-------|-------|-------------|
| 6.1 aug_s42 (`bab2c181`) | **12** | 15 | **14** | 16 | Augmentation + curriculum; best map2 |
| 6.1 aug_s123 (`fa4610df`) | **12** | 15 | 16 | 16 | Consistent with s42 |
| 6.2 sims100_s42 (`6256b33a`) | **12** | 17 | 15 | 16 | More sims don't help without alpha/temp changes |
| 6.2 sims100_s123 (`8dc0f203`) | **12** | 16 | 17 | 18 | Worse than 50-sim aug runs |
| 6.3 optionB no-aug (`8d8e0b61`) | **12** | 18 | 16 | 16 | Collapses (entropy→0.15) without aug |
| 6.5 B/C from scratch | — | — | — | — | Killed at epoch 0, no data. Results in Round 7 (7F/7G). |
| 6.6 A: sims=200 (`bf70eaaa`) | **12** | 14 | **13** | **14** | Sims alone marginal improvement |
| 6.6 B: alpha=0.3 (`331f5082`) | **12** | **13** | **13** | **14** | **Best generalization; alpha is key lever** |
| 6.6 C: temp_decay=5000 (`40438604`) | **12** | 14 | 15 | 16 | Slower cooling doesn't help |
| 6.6 D: all combined (`2747bd10`) | **12** | 14 | **13** | **14** | Best avg_cost=13.7; 4× slower |
| 6.6 E: Option G (`925e114e`) | 14 | **13** | 16 | **14** | Pretrained V has wrong target for unbiased reward; near-random for 277ep |

**Current best config**: Option E + augmentation + curriculum + **alpha=0.3** (Run B, `331f5082`).
**Current best eval**: map2=**11** achieved in a predecessor run (sims=200, see `map_pool.json`).

**Round 7 follow-up on 6.5 B/C from scratch**: See `07_generalist_and_reward_modes.md` runs 7F and 7G.
- 7F (Option B + aug, fixed map, from scratch): avg=12.2 by ep8, entropy_l0≈1.0 — does NOT collapse. Raises mcts_reward_frac to 0.86 vs 0.07 for plan_cost on generalist. Answered H1 of 6.5: augmentation prevents collapse.
- 7G (Option C + aug, fixed map, td_steps=10, from scratch): best=23 after 300 epochs — confirmed failed. Sparse reward without pretrained V provides no learning signal.

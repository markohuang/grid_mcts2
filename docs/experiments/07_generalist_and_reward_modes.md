# Round 07: Generalist Training and Reward Mode Comparison

## Questions

1. Can a pure generalist (random_board) with alpha=0.3 achieve avg_cost < 12 on novel 5×5 maps?
2. Does sims=200 meaningfully help a generalist over sims=50?
3. Does progressive curriculum (sims=200) outperform pure generalist?
4. Does Option G (plan_cost_unbiased) trained from scratch match or beat Option E?
5. How do corrected Option B (layer_delta) and Option C (layer_completion) compare to Option E on a fixed map, all from scratch with identical settings?

## Design

**All runs**: from scratch (no checkpoint), `aug=True`, `alpha=0.3`, `seed=42`, map2 (5×5, 12q, 3 layers), 300 epochs.

**Main runs (7A–7D)**: generalist goal — eval pool pre-populated with 3 random maps via `curriculum_initial_phase=3` so generalization is measurable from epoch 1.

**Side runs (7E–7G)**: reward mode shootout on fixed map2. Only variable is `reward_mode`. Direct comparison to each other and to `ef54adc7` (Option E, alpha=0.1, from scratch) as historical reference.

## Commands

```bash
# 7A: Pure generalist, sims=50, plan_cost
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 --config.random_board=True \
  --config.training.epochs=300 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.experiment.early_stopping_patience=150 \
  --config.experiment.curriculum_maps=3 \
  --config.experiment.curriculum_initial_phase=3

# 7B: Pure generalist, sims=200, plan_cost
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 --config.random_board=True \
  --config.training.epochs=300 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 --config.mcts.num_simulations=200 \
  --config.training.data_augmentation=True \
  --config.experiment.early_stopping_patience=150 \
  --config.experiment.curriculum_maps=3 \
  --config.experiment.curriculum_initial_phase=3

# 7C: Curriculum, sims=200, plan_cost (progressive map expansion)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 \
  --config.training.epochs=300 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 --config.mcts.num_simulations=200 \
  --config.training.data_augmentation=True \
  --config.experiment.early_stopping_patience=100 \
  --config.experiment.curriculum_maps=3 --config.experiment.curriculum_patience=40 \
  --config.training.fixed_map_fraction=0.2

# 7D: Pure generalist, sims=50, plan_cost_unbiased (Option G from scratch)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 --config.random_board=True \
  --config.training.epochs=300 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.experiment.early_stopping_patience=150 \
  --config.env.reward_mode=plan_cost_unbiased \
  --config.experiment.curriculum_maps=3 \
  --config.experiment.curriculum_initial_phase=3

# 7E: Option E from scratch, fixed map2, alpha=0.3 (side: reward mode baseline)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 \
  --config.training.epochs=300 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.experiment.early_stopping_patience=150

# 7F: Option B (layer_delta) from scratch, fixed map2, alpha=0.3 (side)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 \
  --config.training.epochs=300 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.experiment.early_stopping_patience=150 \
  --config.env.reward_mode=layer_delta

# 7G: Option C (layer_completion) from scratch, fixed map2, alpha=0.3, td_steps=10 (side)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 \
  --config.training.epochs=300 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.experiment.early_stopping_patience=150 \
  --config.env.reward_mode=layer_completion \
  --config.training.td_steps=10
```

## Run IDs

| Run | ID | Resumed IDs | Status |
|-----|----|-------------|--------|
| 7A generalist sims=50 | `d90b67b3` | `5e1bc1f0` → `74ee1c9d` | Stopped (diagnostic data collected) |
| 7B generalist sims=200 | `f3760a61` | `5a3b7203` → `5d52140e` | Stopped (diagnostic data collected) |
| 7C curriculum sims=200 | `f351a729` | `c8306b7e` → `35f9f1c0` | Stopped (diagnostic data collected) |
| 7D generalist Option G | `d17e0fac` | `01cfb73b` → `62702890` | Stopped (diagnostic data collected) |
| 7E Option E scratch | `aa0b9487` | `418e3040` | Stopped (mode collapse confirmed ep150) |
| 7F Option B scratch | `21c8bb47` | `ef1434d1` → `da62059a` | Stopped (diagnostic data collected) |
| 7G Option C scratch | `dfcfae68` | — | Stopped (ep300, confirmed failed) |

Diagnostic metrics (cost_hist, val_cost_corr, entropy_layer0/1/2, train_grad_norm) added at final resume point (`74ee1c9d`, `5d52140e`, `35f9f1c0`, `62702890`, `da62059a`).

## Hypotheses

- **H1**: Pure generalist (7A) generalizes better than curriculum (7C) for novel maps due to broader training distribution.
- **H2**: sims=200 (7B) closes the gap to <12 on random maps at the cost of 3.5× compute.
- **H3**: Option G (7D) from scratch eventually matches Option E once V_θ converges to unbiased target, and generalizes better across maps.
- **H4**: Option B (7F) with aug from scratch achieves comparable best_cost to Option E (7E) but converges slower.
- **H5**: Option C (7G) with td_steps=10 converges more slowly than B/E but reaches a similar floor.

## Results

### Diagnostic data summary (10 epochs from resumed runs, 20 games/epoch)

| Run | avg_cost | entropy_l0 | entropy_l1 | entropy_l2 | mcts_reward_frac | avg_mcts_depth | val_cost_corr |
|-----|----------|-----------|-----------|-----------|-----------------|----------------|---------------|
| 7A sims=50 generalist | 17.8–19.0 | 0.21–0.27 | 0.21–0.25 | 0.25–0.30 | 0.07–0.09 | 9.7–10.0 | ≈0 (noisy) |
| 7B sims=200 generalist | 15.6–17.1 | 0.61–0.71 | 0.34–0.44 | 0.34–0.43 | 0.16–0.21 | 10.7–11.0 | ≈0 (noisy) |
| 7C curriculum sims=200 | ep1 only: best=11, avg=13.9 | 0.20 | 0.29 | 0.34 | 0.39 | 11.4 | -0.21 |
| 7D Option G generalist | 17.2–19.4 | 1.9–2.1 | 1.3–1.5 | 1.7–1.9 | 1.0 | 3.3–3.5 | ≈0 |
| 7F Option B fixed map | 12.2–12.8 | 1.00–1.12 | 0.71–0.86 | 1.51–1.58 | 0.86 | 4.5–4.9 | ≈0 (noisy) |

### Hypothesis outcomes

**H1 (7A vs 7C)**: Inconclusive — 7C only produced 1 epoch (avg=13.9 with sims=200) vs 7A's 17.8–19.0 (sims=50). sims=200 accounts for most of the gap. Not enough data on the generalist vs curriculum effect.

**H2 (sims=200 helps)**: Partially confirmed. 7B (sims=200) outperforms 7A (sims=50): avg 15.6–17.1 vs 17.8–19.0, and 7B found best=10. But avg stays 15–17 — sims=200 alone doesn't achieve avg < 12 on random maps.

**H3 (Option G from scratch)**: **Refuted**. 7D shows entropy ≈ 2.0 (essentially random policy) for all 7 diagnostic epochs. mcts_reward_frac=1.0 but avg_mcts_depth=3.3 (far too shallow to guide a 24-step episode). The unbiased reward does give signal at every step, but the value function hasn't bootstrapped and random exploration isn't producing learning. This run is not converging on the timescale of 300 epochs.

**H4 (Option B comparable to Option E)**: **Confirmed and exceeded**. 7F (Option B, layer_delta) achieves avg=12.2–12.8 by ep8 with cost_hist 75% at cost=12. Entropy stays healthy (l0≈1.0, l2≈1.5) — crucially, it does NOT entropy-collapse the way 7E did (ent_l0=0.065 by ep150). Option B maintains exploration while converging. Converges faster than Option E and without collapse.

**H5 (Option C)**: **Confirmed failed**. Sparse reward with no pretrained V = no learning at all. best=23 after 300 epochs.

### mcts_reward_frac: what it measures and what it means

`mcts_reward_frac` = fraction of MCTS simulations where the **net cumulative reward over the traversal path** is |sum| > 1e-6.

Important: individual step rewards under `plan_cost` are almost always non-zero (nearly every atom move changes count_groups outcomes). What mcts_reward_frac is measuring is whether the **sum** over ~10 steps is non-zero. For `plan_cost`, step rewards are signed (positive when move reduces remaining cost, negative when it worsens it). On novel random maps where the policy doesn't know the optimal arrangement, roughly equal proportions of moves help and hurt → rewards cancel across the path → |sum| ≈ 0.

For `layer_delta`, the layer-completion reward telescopes to `-actual_layer_cost` (always negative, always non-zero). Any simulation that crosses a layer boundary gets a guaranteed non-zero sum — which is why mcts_reward_frac=0.86 for 7F largely reflects "fraction of simulations reaching a layer boundary."

Values observed:
- **7A (plan_cost, sims=50): 0.07–0.09** — rewards cancel on novel maps; individual UCB step rewards still non-zero but net path signal is small
- **7B (plan_cost, sims=200): 0.16–0.21** — modestly better; avg_cost stays 15–17 despite best=10 being found occasionally
- **7F (layer_delta, fixed): 0.86** — layer completion guaranteed non-zero; fast convergence to avg=12.2

Note: even when mcts_reward_frac is low, MCTS is not literally blind — individual step rewards ARE stored in `child.reward` and used in UCB. The low value indicates reward path cancellation on novel maps, not total absence of signal. The **causal link** from mcts_reward_frac to avg_cost is a correlation, not proven causation.

### Value function calibration

`val_cost_corr` is near-zero for all runs with high variance (±0.5 per epoch at n=20 games). This metric is **too noisy to be conclusive** at n=20. A true correlation of 0.3 could easily appear as -0.5 to 0.8 over 20 samples. We cannot confirm H3 (value miscalibration) from this data — val_cost_corr needs either more games per epoch or aggregation across many epochs to be informative. 7F converging well despite near-zero val_cost_corr suggests the policy network is the primary MCTS driver, but V_θ calibration status remains unclear.

### Option B entropy vs Option E collapse

7E (plan_cost, fixed) collapsed to ent_l0=0.065 by ep150 — policy deterministically committed to cost=12. 7F (layer_delta, fixed) maintains ent_l0≈1.0 for 10 diagnostic epochs even while converging to avg=12.2. The frequent reward signal from layer_delta prevents premature commitment to suboptimal actions.

### H1/H2/H3 status after diagnostics

- **H1 (bootstrap variance)**: Untested directly. No clear signal in available metrics.
- **H2 (mode collapse)**: Confirmed for 7E (ent_l0=0.065, std=0.00, grad≈0.25). Not observed for 7F (layer_delta), suggesting reward mode affects collapse tendency.
- **H3 (value miscalibration)**: Inconclusive — val_cost_corr too noisy at n=20.
- **Observed but unconfirmed**: plan_cost shows reward path cancellation on novel maps (mcts_reward_frac 0.07); layer_delta avoids this by construction. Whether this is a primary bottleneck or a consequence of policy quality remains to be determined.

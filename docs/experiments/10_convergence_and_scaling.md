# Round 10: Convergence Studies and Scaling Indicators

## Decision

Round 08 showed that `layer_delta + plan_cost_search_bonus_weight=2.0` was transformative on a fixed map (avg 12.0 vs 17.8 control). Round 09 is testing whether this survives on generalist. Regardless of R09 outcome, we need long runs to characterize convergence behavior and assess whether the approach is worth scaling to HPC.

## Goal

Produce convergence curves (cost vs compute) on fixed and generalist setups to answer:

1. Does cost monotonically decrease with more training, or does it plateau / regress?
2. Is the current NN (139K params) underfitting? Does 3× capacity help?
3. Does more self-play data per epoch (50 vs 20 games) improve convergence or alleviate mode collapse?
4. What does the cost distribution look like at convergence — tight around optimal, or bimodal/spread?

Each fixed-map run to convergence is treated as **one datapoint**: for map X with config C, the converged cost is Y. If these datapoints show near-optimal convergence, the approach is worth scaling to HPC (parallel self-play across many maps, H100 training for a generalist).

## Scalability Indicators to Track

Already in the logging (via R08 instrumentation):

- `cost_hist` — cost distribution narrows toward optimal?
- `cumulative_time` — cost vs wall-time curve
- `entropy_layer0/1/2` — no collapse during extended training?
- `train_grad_norm` — learning stall detection
- `val_cost_corr` — value calibration improving over time?
- `mcts_reward_abs_sum_mean` — raw per-step reward magnitude (non-cancelling)
- `mcts_boundary_reach_frac` — are simulations crossing layer boundaries?
- `mcts_sign_changes_mean` — reward oscillation
- `avg_mcts_depth` — search depth trajectory

**New indicators to watch:**
- Cost vs total games played (sample efficiency)
- Epoch at which cost first reaches each threshold (10, 11, 12)
- Final cost_std at convergence (tight = reliable, spread = search-dependent)

## Design

All runs: `alpha=0.3`, `aug=True`, `seed=42`, map2 (5×5, 12q, 3 layers), from scratch.

Best config from R08: `layer_delta` + `plan_cost_search_bonus_weight=2.0`.

### Experiments

| Run | Map | games/ep | NN | search_bonus | Epochs | Purpose |
|-----|-----|----------|----|----|------|---------|
| 10A | fixed map2 | 50 | current (139K) | 2.0 | 500 | Fixed-map convergence baseline |
| 10B | fixed map2 | 50 | 2x (418K, v=128 p=64) | 2.0 | 500 | NN capacity on fixed map |
| 10C | generalist | 50 | current (139K) | 2.0 | 500 | Generalist convergence |
| 10D | generalist | 50 | 2x (418K, v=128 p=64) | 2.0 | 500 | NN capacity for generalist |
| 10E | fixed map2 | 100 | current (139K) | 2.0 | 500 | Data volume: 2× games/epoch |

### Comparisons

- **10A vs 10B**: does 3× NN capacity improve fixed-map floor (below cost 12)?
- **10C vs 10D**: does capacity matter more for generalist (higher diversity, harder learning)?
- **10A vs 10C**: fixed vs generalist gap with matched config and long training
- **10A vs 10E**: does 2× data volume push convergence further or just speed it up?
- **All**: cost vs cumulative_games curves for sample efficiency

### Timing (measured, all 5 running concurrently on 2×A6000 + 48 CPU)

- 10A (50 games, 139K): ~160s/epoch → ~22 hours
- 10B (50 games, 418K): ~120s/epoch → ~17 hours
- 10C (50 games, 139K, generalist): ~110s/epoch → ~15 hours
- 10D (50 games, 418K, generalist): ~130s/epoch → ~18 hours
- 10E (100 games, 139K): ~250s/epoch → ~35 hours

Times include search_bonus overhead (remaining_cost computation at every MCTS sim step).

Note: `plan_cost_search_bonus_weight > 0` auto-enables `track_plan_delta=True`, which computes `_compute_remaining_cost()` at every MCTS sim step even under `layer_delta`. This adds overhead but provides the cross-layer signal in search without contaminating the value target.

## Commands

```bash
# 10A: Fixed map, current NN, 50 games/ep
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 \
  --config.training.epochs=500 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.experiment.early_stopping_patience=200 \
  --config.experiment.study=round10_convergence_scaling \
  --config.experiment.hypothesis=baseline \
  --config.experiment.variant=fixed_50g_139k \
  --config.experiment.tags=convergence,fixed,specialist \
  --config.experiment.notes="fixed map2, 50 games/ep, current NN, long run to convergence"

# 10B: Fixed map, 2x NN, 50 games/ep
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 \
  --config.training.epochs=500 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.network.v_hsize=128 --config.network.p_hsize=64 \
  --config.experiment.early_stopping_patience=200 \
  --config.experiment.study=round10_convergence_scaling \
  --config.experiment.hypothesis=capacity \
  --config.experiment.variant=fixed_50g_418k \
  --config.experiment.tags=convergence,fixed,specialist,capacity \
  --config.experiment.notes="fixed map2, 50 games/ep, 2x NN (418K), capacity test"

# 10C: Generalist, current NN, 50 games/ep
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 --config.random_board=True \
  --config.training.epochs=500 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.experiment.early_stopping_patience=200 \
  --config.experiment.curriculum_maps=3 \
  --config.experiment.curriculum_initial_phase=3 \
  --config.experiment.study=round10_convergence_scaling \
  --config.experiment.hypothesis=generalist \
  --config.experiment.variant=generalist_50g_139k \
  --config.experiment.tags=convergence,generalist \
  --config.experiment.notes="generalist, 50 games/ep, current NN, long run convergence study"

# 10D: Generalist, 2x NN, 50 games/ep
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 --config.random_board=True \
  --config.training.epochs=500 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.network.v_hsize=128 --config.network.p_hsize=64 \
  --config.experiment.early_stopping_patience=200 \
  --config.experiment.curriculum_maps=3 \
  --config.experiment.curriculum_initial_phase=3 \
  --config.experiment.study=round10_convergence_scaling \
  --config.experiment.hypothesis=generalist_capacity \
  --config.experiment.variant=generalist_50g_418k \
  --config.experiment.tags=convergence,generalist,capacity \
  --config.experiment.notes="generalist, 50 games/ep, 2x NN (418K), capacity + convergence"

# 10E: Fixed map, current NN, 100 games/ep (data volume)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 \
  --config.training.epochs=500 --config.training.num_selfplay=100 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.experiment.early_stopping_patience=200 \
  --config.experiment.study=round10_convergence_scaling \
  --config.experiment.hypothesis=data_volume \
  --config.experiment.variant=fixed_100g_139k \
  --config.experiment.tags=convergence,fixed,specialist,data-volume \
  --config.experiment.notes="fixed map2, 100 games/ep (2x data volume), convergence test"
```

## Run IDs

| Run | ID | Params | Epochs | Status |
|-----|----|--------|--------|--------|
| 10A fixed 50g 139K | `f91ba571` | 139K | 214 | Completed |
| 10B fixed 50g 418K | `48c71be4` | 418K | 265 | Completed |
| 10C generalist 50g 139K | `bf5f8afd` | 139K | 202 | Completed |
| 10D generalist 50g 418K | `565922d3` | 418K | 201 | Completed |
| 10E fixed 100g 139K | `7a8d9d8f` | 139K | 249 | Completed |

## Decision Rules

### If 10A converges to cost ≤ 11 on fixed map

The approach works for specialists. Proceed with:
- HPC parallel self-play across many maps
- Aggregation / distillation into a generalist

### If 10B (2x NN) materially beats 10A

Current NN is underfitting. Scale up NN for all future runs. If 10D also beats 10C, capacity is the bottleneck for generalist too.

### If 10A plateaus at cost 12 despite 500 epochs

The floor is structural (search limitation, not capacity). Next steps:
- More sims (200+)
- Better search heuristics
- SMT-guided pre-training for policy warm-start

### If 10C/10D generalist avg ≤ 14 with tight cost_std

Generalist training is viable at this scale. Worth investing in HPC scale-up.

### If generalist plateaus at avg ≥ 17 regardless of NN size

Fundamental generalist bottleneck. Consider:
- Distilling from per-map specialists instead of end-to-end generalist training
- Curriculum with longer patience
- Architectural changes (attention, transformer)

## Results

### Summary table

| Run | avg_cost | cost_hist | eval_map0_avg | ent_l0 | depth | grad | val_corr | Wall time |
|-----|----------|-----------|---------------|--------|-------|------|----------|-----------|
| 10A fixed 139K | **11.0** | {11: 100%} | 11.0 | 0.60 | 6.9 | 0.42 | 0.0* | 13.3h |
| 10B fixed 418K | 12.0 | {12: 100%} | 12.0 | 1.05 | 5.0 | 0.38 | 0.0* | 14.7h |
| 10C gen 139K | 15.2 | {12-19, spread} | 12.0 | 1.52 | 4.4 | 13.6 | 0.35 | 11.1h |
| 10D gen 418K | 15.4 | {12-19, spread} | 15.0 | 1.54 | 4.4 | 6.4 | 0.28 | 11.1h |
| 10E fixed 100g 139K | **11.0** | {11: 100%} | 11.0 | 0.76 | 5.7 | 0.41 | 0.0* | 22.6h |

*val_corr=0.0 on fixed-map runs is trivial: all games have the same cost → no variance to correlate.

### Convergence trajectories

**10A (fixed 139K)**: Reached avg=11.0 by **epoch 17** (1.0 hour). 100% of games at cost=11 from then on. Stayed locked for 200+ more epochs — perfect convergence.

**10B (fixed 418K)**: Hit best=11 at epoch 1, but converged to cost=12 (100%) by epoch 76. Never broke below 12 in 265 epochs. **Larger NN converged to a worse attractor.**

**10E (fixed 100g 139K)**: Same floor as 10A (cost=11, 100%). More data volume didn't push below 11.

**10C (generalist 139K)**: Reached avg=15.0-15.3 by epoch 5 and **plateaued for 200 epochs**. eval_map0_best=12 (can solve the fixed map in eval). Self-play cost distribution wide (12-19). Grad norms increasing (2→13) — gradient signal exists but doesn't translate to cost improvement.

**10D (generalist 418K)**: Same plateau as 10C (avg=15.4). No benefit from 3× capacity.

### Key findings

**1. Fixed-map specialist works: cost=11, 100% convergence.**

The 139K model with layer_delta + search_bonus=2.0 converges to cost=11 on map2 within 17 epochs (1 hour). This is one step from the per-layer SMT solver's cost=10. The approach works for specialists and is worth scaling.

**2. Larger NN hurts on fixed map.**

10B (418K) locked at cost=12, one level ABOVE 10A's cost=11. The larger model converged to a different, worse attractor. Possible cause: more parameters make the loss landscape smoother, allowing the optimizer to settle into a broader but shallower basin (cost=12) rather than the narrower cost=11 basin. Entropy was higher in 10B (ent_l0=1.05 vs 0.60), consistent with less committed convergence.

**3. Generalist plateaus at avg=15 regardless of NN size or training length.**

10C and 10D both plateau at avg~15 from epoch 5 onwards. 200 epochs of additional training produce zero improvement. This is NOT caused by:
- Mode collapse (entropy ~1.5, healthy)
- Dead gradients (grad_norm 6-13, active)
- Value miscalibration (val_corr=0.28-0.35, positive)
- Insufficient capacity (418K no better than 139K)

The model CAN solve fixed maps in eval (eval0_avg=12) but averages 15 on novel random maps. This is a **generalization gap**: the model learns map-specific strategies that don't transfer.

**4. 100 games/epoch doesn't break the specialist floor.**

10E matches 10A at cost=11. More data per epoch speeds up convergence but doesn't push past the cost=11 floor.

**5. Scalability assessment.**

| Indicator | Fixed map | Generalist |
|-----------|-----------|------------|
| Converges to near-optimal? | Yes (11 vs SMT's 10) | No (plateaus at 15) |
| Mode collapse? | No | No |
| Learning still active? | Stalled (grad=0.4) | Active (grad=13) |
| NN capacity bottleneck? | No (418K worse) | No (418K same) |
| More data helps? | No (100g same) | Untested |

### Implications for HPC scale-up

The specialist result is strong: train one model per map → cost=11 in 1 hour. This is parallelizable across maps. But the generalist result shows that end-to-end generalist training on random maps does not work at this scale — the model plateau at avg=15 despite active gradients and healthy entropy.

Two paths forward:
1. **Specialist distillation**: Train N specialists in parallel on HPC (each converges in ~1h), then distill their knowledge into a single generalist via behavioral cloning or policy aggregation.
2. **Prior learning / pre-training**: Use SMT solutions or specialist policies as training data to warm-start a generalist, then fine-tune with MCTS.

---

## Phase 2: Sequential Specialist Training

### Motivation

End-to-end generalist training (10C/10D) plateaus at avg=15. But the specialist converges to 11 in 17 epochs. Can we bridge this gap by training the same model on maps one at a time, each time loading from the previous phase's checkpoint?

Each map's convergence trajectory is one "data sample" in a meta-learning sense. After N maps, the conflicting map-specific gradients should cancel out, and the model should retain the generalizable features. The 139K model can't memorize 100+ maps — it's FORCED to generalize.

### Design

Start from 10A's converged specialist checkpoint (`f91ba571`, cost=11 on map2).

For each phase k = 1..20:
- Set `random_board=True, random_board_seed=k*100` → all games use the SAME random map (deterministic from seed)
- Train for 30 epochs (early_stopping_patience=15)
- Load the resulting checkpoint into the next phase

All other settings match 10A: layer_delta, search_bonus=2.0, 50 games/ep, alpha=0.3, aug=True.

Script: `experiments/run_sequential_specialist.sh`

### Success criteria (not wishful thinking)

| Metric | What to measure | Success signal | Failure signal |
|--------|----------------|----------------|----------------|
| Starting avg_cost | avg_cost at epoch 1 of each phase | Decreases over phases (15 → 13 → 12) | Stays flat at 15+ |
| Convergence speed | Epochs to reach cost ≤ 12 per phase | Decreases (17 → 10 → 5) | Stays flat at 17+ |
| Map2 retention | eval_map0_best_so_far | Stays at 11-12 | Degrades to 15+ (catastrophic forgetting) |
| Generalist probe | After phase 20, run with seed=-1 (random maps) | avg < 14 | avg ≥ 15 (same as 10C) |

### Results

| Phase | seed | start_avg | final_avg | best | epochs | eval_map0 | first ≤12 |
|-------|------|-----------|-----------|------|--------|-----------|-----------|
| 1 | 100 | 15.3 | 12.0 | 12 | 16 | 11 | ep6 |
| 2 | 200 | 16.7 | 13.0 | 11 | 16 | 11 | ep13 |
| 3 | 300 | 15.9 | 12.0 | 12 | 30 | 14 | ep25 |
| 4 | 400 | 13.8 | 12.3 | 12 | 16 | 16 | — |
| 5 | 500 | 16.3 | 11.8 | 11 | 24 | 14 | ep8 |
| 6 | 600 | 17.1 | 14.0 | 14 | 25 | 13 | — |
| 7 | 700 | 16.7 | 16.0 | 16 | 16 | 13 | — |
| 8 | 800 | 14.3 | 12.0 | 12 | 19 | 14 | ep4 |
| 9 | 900 | 15.7 | 11.7 | 11 | 17 | 14 | ep7 |
| 10 | 1000 | 18.8 | 14.2 | 13 | 20 | 13 | — |
| 11 | 1100 | 12.7 | 11.0 | 11 | 22 | 13 | ep4 |
| 12 | 1200 | 15.6 | 14.0 | 14 | 26 | 12 | — |
| 13 | 1300 | 15.9 | 13.1 | 13 | 22 | 12 | — |
| 14 | 1400 | 18.6 | 16.8 | 16 | 16 | 12 | — |
| 15 | 1500 | 16.6 | 15.0 | 15 | 20 | 12 | — |
| 16 | 1600 | 15.1 | 11.7 | 11 | 25 | 13 | ep10 |
| 17 | 1700 | 15.2 | 13.0 | 13 | 16 | 12 | — |
| 18 | 1800 | 15.9 | 12.1 | 12 | 20 | 12 | — |
| 19 | 1900 | 14.5 | 14.0 | 14 | 17 | 13 | — |
| 20 | 2000 | 14.6 | 13.0 | 13 | 18 | 13 | — |

### Analysis

- **Starting cost**: no downward trend over 20 phases (range 12.7–18.8). No evidence of learning-to-learn.
- **Convergence**: 8/20 phases reached avg ≤ 12. Varies by map difficulty, not by phase number.
- **Map2 retention**: degraded from 11 (phase 1-2) to 12-13 (phases 12-20). Partial retention, not catastrophic forgetting, but not building a shared representation either.
- **Conclusion**: sequential specialist training does NOT build transferable features. Each map is solved essentially from scratch. The model forgets previous maps while learning new ones.

Timing: ~1 hour per phase × 20 phases ≈ 20 hours total (sequential).

### After sequential training: generalist probe

If convergence speed improves, run the final checkpoint with `random_board=True, random_board_seed=-1` (truly random maps) for 100 epochs. This tests whether the sequential training produced a genuinely better generalist prior.

### Future: data distillation (approach 1)

If sequential training shows positive transfer, the next step is proper distillation:
1. Train N specialists in parallel (N=50-100 maps, each ~1h on HPC)
2. Collect converged games: (state, MCTS_policy, game_cost) tuples
3. Pre-fill replay buffer with multi-map specialist data
4. Train generalist from scratch on this buffer + ongoing self-play

This requires code changes (replay buffer pre-fill from disk) but is the cleaner HPC-scale approach. Only worth building if sequential training confirms transferable features exist.

---

## Phase 3: Architecture and Representation Analysis

### Why generalization fails — feature representation gaps

Tracing the full information flow through the network reveals five structural issues:

**1. Gate pairs are not encoded — only participation flags.**

`get_features()` adds `+1` to a qubit's column if it participates in a layer. Gates `[(3,5),(7,9)]` and `[(3,7),(5,9)]` produce IDENTICAL features, despite requiring completely different atom arrangements. The network cannot compute the cost-determining distance between PAIRED qubits because it doesn't know who pairs with whom.

**2. No spatial structure.**

The 5×5 grid is flattened to 25 cells. The network has no inductive bias for row/column adjacency. Parallel grouping depends on whether moves cross in rows/columns — the most important structural constraint is invisible at the feature level.

**3. No progress information.**

`tasks_done`, `current_atom_idx`, and remaining atoms are not in the features. The network must infer phase from which gate indicators are present/absent.

**4. Qubit identity is additive.**

`x = x + qubit_emb` competes with board state in the same vector space. On fixed maps, the board signal dominates and the identity signal gets suppressed.

**5. MLPMixer mixes all qubits equally.**

Conv1d patch mixing uses fixed weights — no mechanism to attend specifically to gate partners. Transformer attention would naturally learn "look at my gate partner."

### Why this explains specialist-works-but-generalist-fails

On a fixed map, gate pairs never change. The network memorizes "for THIS set of pairs, place atoms in THESE cells." It doesn't need to compute pairwise distances or understand spatial constraints — just a lookup table.

On random maps, pairs change every game. The network must understand "qubit 3 pairs with qubit 5, they're far apart, move 3 closer." But the features don't encode the pairing, and the architecture has no spatial inductive bias to help. The network is trying to generalize with fundamentally inadequate input representation.

### Diagnostics to add (before changing architecture)

1. **Per-module gradient norms**: Measure gradient norm for mixer1, mixer2, qubit_emb, and heads separately. If mixer2 gradients vanish, the task-mixing stage isn't learning.

2. **Feature ablation**: Zero out the gate indicator slots (slots 1..N) and re-evaluate. If performance barely changes, the network isn't using task structure at all — just board state memorization.

3. **Pairwise distance baseline**: Compute the sum of Manhattan distances between all gate-paired qubits. Correlate with game cost. If this is highly predictive, it confirms the network is missing the key feature.

### Proposed feature improvements (minimal arch change)

**A. Add pairwise gate features:**
Instead of just gate participation flags, encode pair-relative positions:
- For each layer, for each gate pair (q1, q2): encode the relative position (Δrow, Δcol) between q1 and q2 as additional channels
- Or: add a pairwise distance channel per layer — `pair_dist[cell, qubit]` = Manhattan distance to qubit's gate partner(s) in that layer

**B. Add spatial coordinates:**
Append (row/board_h, col/board_w) normalized coordinates as additional channels per cell. This gives the network row/column awareness without architecture changes.

**C. Add progress features:**
Append `tasks_done / num_tasks` and `current_atom_idx / len(relevant_atoms)` as scalar channels broadcast across the feature tensor.

**D. Replace MLPMixer with multi-head self-attention (over qubits):**
This lets the network learn to attend to gate partners rather than mixing all qubits equally. A 2-layer transformer over the qubit dimension with dim=64 is comparable in parameter count to the current mixer.

### Priority order

1. **Diagnostics** (zero risk, high information): per-module gradients, feature ablation, pairwise distance correlation
2. **Feature improvements A-C** (low risk, medium effort): test whether better features improve generalist performance with the existing architecture
3. **Architecture change D** (medium risk, medium effort): if better features aren't enough, the mixing mechanism itself needs to change

### Architecture validation results (arch_sanity_check.py)

Supervised cost prediction on 50K random 5×5 maps (×8 augmentation = 400K samples), train/test split BEFORE augmentation (no leakage), 300 epochs.

| # | Architecture | Features | Acc | Pearson r |
|---|-------------|----------|-----|-----------|
| 1 | MLPMixer | current (flat participation flags) | 0.329 | 0.013 |
| 2 | MLPMixer | classifier (cross-positional pairs) | 0.379 | 0.464 |
| 3 | Transformer | current (flat participation flags) | 0.337 | 0.023 |
| **4** | **Transformer** | **classifier (cross-positional pairs)** | **0.889** | **0.954** |
| 5 | MLPMixer | pairwise distances | 0.345 | 0.339 |
| 6 | Transformer | pairwise distances | 0.512 | 0.721 |

**Context**: The original rewards_classifier_mlp.py achieved 77% val accuracy on 3×3 boards (9 cells, 9 qubits) using MLPMixer + cross-positional pair features. At 5×5 scale, the same MLPMixer + same features only reaches 37.9% — the fixed Conv1d mixing in MLPMixer cannot handle the pairwise combinatorics at larger scale.

**Conclusions**:

1. **Both ingredients are necessary.** Transformer alone (r=0.023) and classifier features alone with MLPMixer (r=0.464) are insufficient. Together: r=0.954.
2. **The feature encoding was the primary bottleneck.** Current env's flat flags (`+1 to all cells for participating qubits`) destroy gate pairing info. The classifier encoded pair structure as cross-positional markers (`feat[cell_of_q1, q2] = 1`).
3. **Attention over qubits is required at 5×5 scale.** MLPMixer's fixed Conv1d cannot learn conditional pairwise interactions. Self-attention naturally discovers gate partners.
4. **Pairwise distances are a lossy summary (r=0.721 vs r=0.954).** Providing raw pair structure lets the network learn its own distance-like representations.

**Action**: Restore cross-positional pair encoding in `env.get_features()` and replace MLPMixer with self-attention over qubits in ValueNetwork/PolicyNetwork.

### Changes implemented

**Feature encoding** (`env.py:get_features()`, `network.py:make_features()`):
- OLD: flat participation flags — `feat[:, q] += 1.0` for each qubit q in a layer (broadcasts to all cells)
- NEW: cross-positional pair markers — `feat[cell_of_q1, q2] = 1.0` for each gate pair (q1, q2)
- Implemented via `_board_feat @ _gate_pair_matrix[t]` (matrix multiply for speed in MCTS hot path)
- Encodes both WHO pairs with whom and WHERE they are spatially

**Architecture** (`network.py: ValueNetwork, PolicyNetwork`):
- OLD: MLPMixer (fixed Conv1d patch mixing over qubits) — 139K params
- NEW: TransformerEncoder (self-attention over qubits) — 298K params
- Stage 1: `proj_in → qubit_emb + TransformerEncoder` (qubits attend to gate partners)
- Stage 2: `task_proj → TransformerEncoder` (cross-task/cross-layer reasoning)
- All 41 existing tests pass; FakeNet and real-network smoke tests pass

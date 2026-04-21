# Round 11: Architecture Validation (Transformer + Cross-Positional Features)

## Context

Round 10 Phase 3 identified that the MLPMixer + flat participation flags cannot learn cost prediction on 5×5 maps (Pearson r=0.013 on held-out test set). Replacing with Transformer + cross-positional pair features achieves r=0.954 on the same supervised task.

Changes implemented:
- **Features**: `feat[cell_of_q1, q2] = 1.0` for each gate pair (q1, q2) — encodes pairing AND spatial layout
- **Architecture**: MLPMixer → TransformerEncoder (self-attention over qubits, then over tasks)
- Param count: 298K (was 139K)

## Goal

Validate the new architecture against old results on the same reward modes. The question is NOT "which reward mode is best" (already answered: layer_delta + search_bonus). The question is: **does the new architecture improve generalist training where the old one plateaued?**

## Design

Benchmark against Round 10 results. Same configs, only the architecture changed.

All runs: map2 (5×5), seed=42, alpha=0.3, aug=True, 50 games/epoch.

### Reward modes to test

Three active modes (see `docs/reward_modes.md`):

| Label | Config | Why test |
|-------|--------|---------|
| **layer_delta + search bonus** | `reward_mode=layer_delta`, `search_bonus=2.0` | Best config from R10 — does the new arch push past cost=11? |
| **plan_cost** | `reward_mode=plan_cost` | Baseline — did the old arch's plan_cost generalist plateau (avg=17) because of features? |
| **layer_completion** | `reward_mode=layer_completion`, `td_steps=10` | Failed from scratch with old arch (R07: best=23). Worth retesting — better V_θ from transformer may bootstrap faster. |

### Experiments

| Run | Map | Reward | Epochs | Benchmark (old arch) |
|-----|-----|--------|--------|---------------------|
| 11A | fixed | layer_delta + bonus | 200 | R10 10A: avg=11.0 @ep17 |
| 11B | generalist | layer_delta + bonus | 300 | R10 10C: avg=15.2 (plateaued) |
| 11C | generalist | plan_cost | 300 | R10 10C equivalent with plan_cost |
| 11D | fixed | layer_completion (td=10) | 200 | R07 7G: best=23 (failed) |

## Commands

```bash
# 11A: Fixed map, layer_delta + search bonus (benchmark vs R10 10A)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 \
  --config.training.epochs=200 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.experiment.early_stopping_patience=100 \
  --config.experiment.study=round11_arch_validation \
  --config.experiment.hypothesis=fixed_specialist \
  --config.experiment.variant=fixed_ld_bonus \
  --config.experiment.tags=arch-validation,fixed,layer-delta,search-bonus \
  --config.experiment.notes="New arch (transformer+cross-pos), fixed map, layer_delta+bonus, benchmark vs R10 10A"

# 11B: Generalist, layer_delta + search bonus (benchmark vs R10 10C)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 --config.random_board=True \
  --config.training.epochs=300 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.experiment.early_stopping_patience=150 \
  --config.experiment.curriculum_maps=3 \
  --config.experiment.curriculum_initial_phase=3 \
  --config.experiment.study=round11_arch_validation \
  --config.experiment.hypothesis=generalist_ld_bonus \
  --config.experiment.variant=gen_ld_bonus \
  --config.experiment.tags=arch-validation,generalist,layer-delta,search-bonus \
  --config.experiment.notes="New arch, generalist, layer_delta+bonus, benchmark vs R10 10C (old avg=15.2)"

# 11C: Generalist, plan_cost (benchmark old-arch generalist)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 --config.random_board=True \
  --config.training.epochs=300 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.experiment.early_stopping_patience=150 \
  --config.experiment.curriculum_maps=3 \
  --config.experiment.curriculum_initial_phase=3 \
  --config.experiment.study=round11_arch_validation \
  --config.experiment.hypothesis=generalist_plan_cost \
  --config.experiment.variant=gen_plan_cost \
  --config.experiment.tags=arch-validation,generalist,plan-cost \
  --config.experiment.notes="New arch, generalist, plan_cost, benchmark vs old-arch generalist (avg=17.5)"

# 11D: Fixed map, layer_completion, td_steps=10 (retest failed R07 7G)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 \
  --config.training.epochs=200 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=layer_completion \
  --config.training.td_steps=10 \
  --config.experiment.early_stopping_patience=100 \
  --config.experiment.study=round11_arch_validation \
  --config.experiment.hypothesis=layer_completion_retest \
  --config.experiment.variant=fixed_lc_td10 \
  --config.experiment.tags=arch-validation,fixed,layer-completion \
  --config.experiment.notes="New arch, fixed map, layer_completion td=10, retest R07 7G (old best=23)"
```

## Run IDs

| Run | ID | Params | Status |
|-----|----|--------|--------|
| 11A fixed ld+bonus | `97596cd9` | 298K | Running |
| 11B gen ld+bonus | `8a4ea7e9` | 298K | Running |
| 11C gen plan_cost | `9e7006fc` | 298K | Running |
| 11D fixed lc td=10 | `777a866f` | 298K | Running |

## Success criteria

| Run | Old arch result | Success threshold | Stretch goal |
|-----|----------------|-------------------|-------------|
| 11A | avg=11.0 @ep17 | avg ≤ 11 in ≤ 20 epochs | avg ≤ 10 (beat old arch) |
| 11B | avg=15.2 (plateau) | avg < 14 (break plateau) | avg ≤ 12 (generalist solved) |
| 11C | avg=17.5 (R09) | avg < 15 (arch helps plan_cost too) | avg ≤ 13 |
| 11D | best=23 (total failure) | best ≤ 15 (learns something) | avg ≤ 12 (sparse reward works) |

## Decision rules

**If 11B breaks below avg=14**: The architecture was the generalist bottleneck. Proceed with transformer + cross-positional features as the default for all future runs and HPC scaling.

**If 11A reaches avg ≤ 10**: The new arch improves even the specialist — potentially matching SMT solver quality.

**If 11D learns (best < 15)**: Layer_completion becomes viable, meaning the simplest reward formulation works when paired with the right architecture.

**If 11B still plateaus at avg ≥ 15**: The generalist problem is not purely representational — need to revisit training signal, data diversity, or curriculum.

## Results

| Run | Epochs | avg_cost | best | eval0_bf | ent_l0 | depth | Old arch |
|-----|--------|----------|------|----------|--------|-------|----------|
| 11A fixed ld+bonus | 102 (early-stop) | 11.0 | 11 | 11 | 0.93 | 5.0 | 11.0 @ep17 |
| 11B gen ld+bonus | 152 (early-stop) | 14.5 | 12 | 11 | 1.27 | 4.5 | 15.2 (plateau) |
| 11C gen plan_cost | 151 (early-stop) | 17.6 | 13 | 12 | 0.30 | 9.5 | 17.5 |
| 11D fixed lc td=10 | 123 (early-stop) | 24.0 | 18 | 16 | 2.50 | 2.3 | 23 |

### Verdict

**The new architecture (transformer + cross-positional features) does NOT solve the generalist problem.**

- **11A**: Matches old arch's cost=11 but takes 5× longer to converge (ep90 vs ep17). The policy transformer at p_hsize=32 is undersized — policy loss stays higher than old arch throughout training.
- **11B**: avg=14.5 vs old 15.2 — a modest 0.7 improvement, NOT the breakthrough to avg<14 we needed. Found best=10 once (proving it's possible) but couldn't do it consistently. Plateau confirmed from ep50 onwards.
- **11C**: Same entropy collapse (0.30) and avg (17.6) as old arch with plan_cost. The new features/arch don't help plan_cost's mode collapse.
- **11D**: Layer_completion still fails from scratch regardless of architecture. Sparse reward can't bootstrap.

### What we learned

1. **The supervised sanity check (r=0.954) doesn't transfer to RL training.** Predicting cost from a static board state (supervised) is different from learning a policy through MCTS self-play. The transformer CAN represent the cost function, but the RL training loop doesn't provide enough signal per map for the generalist to learn it.

2. **The generalist bottleneck is training regime, not architecture.** Each game sees a fresh random map exactly once. The model never gets the deep per-map convergence that specialists need (17+ epochs). The supervised task had 360K samples with augmentation; the RL generalist sees each map for one game.

3. **The policy head at p_hsize=32 is too small for the transformer.** Self-attention with dim=32 gives nhead=2 with 16 dims per head — too few for meaningful attention. This explains the 5× slower convergence on the specialist task.

### Implication for next round

The architecture change is necessary but not sufficient. The real bottleneck is the training regime: every game on a fresh random map gives shallow, noisy signal. Two approaches to test:

1. **Fixed map pool training**: Train on a small fixed pool of 10-20 maps for many epochs. The model sees each map repeatedly (like the specialist) but must generalize across the pool. Middle ground between sequential specialist (1 map, no generalization) and pure random (new map every game, no depth).

2. **Increase p_hsize**: Match policy head to value head size (p_hsize=64) so the transformer has enough capacity for meaningful attention in the policy network.

## Design decision (2026-04-20): bet on data scale

**Assumption.** With (a) the cross-positional representation validated by the supervised task (r=0.954 on held-out), (b) a transformer that self-attends over qubits and tasks, and (c) the new `p_hsize=64` policy capacity, the architecture and features are no longer the bottleneck on 5x5. 11A hit SMT-tight cost=11 on the specialist — the model provably *can* represent the solution. The generalist plateau is a sample-efficiency / regime problem, not a representational one.

**The gap that actually matters.** Our self-play budget has been the first-order limit all along. Comparison with the MCTS frameworks whose design we inherit:

| System | Self-play games (order) | Sims per move | Notes |
|---|---|---|---|
| AlphaDev | ~10^9 moves, 10^7+ games | ~800 | TPU pool for weeks per problem |
| AlphaZero chess | ~44 M games | 800 | 5000 TPUs, 9 h |
| AlphaZero Go | ~29 M games | 1600 | 5000 TPUs, 13 days |
| lc0 | ~10^8+ games (ongoing) | 800 training / 10k tournament | distributed contribution |
| **Ours (R10-R13)** | ~10^3–10^4 games per run | 50–800 | single-process per epoch |
| **Wave01 (HPC one shot)** | **2 × 10^4 games** | 800 | 20-node SLURM array, untrained weights |

Even wave01's 20k games is 3-4 orders of magnitude below AlphaZero and 5+ below AlphaDev, at a *smaller* branching factor (~12 vs ~35-400). Before considering further representational surgery or new inductive biases, we commit to exhausting the data-scale lever.

**Decision.** Default posture for the next round of work: scale self-play throughput, not architecture. Concretely:

1. Run HPC self-play waves in the 10^5 - 10^6 game range per training checkpoint, cycling selfplay → offline training → new checkpoint (AlphaDev loop, not per-epoch inline self-play).
2. Only revisit architecture if a post-scale diagnostic shows the value/policy head failing to fit the collected data (i.e. train loss plateau with held-out cost predictability low) — that would be evidence the inductive bias genuinely caps performance rather than the training signal.
3. Hold the arch stable (transformer + cross-pos + p_hsize=64) across scale-up so checkpoints remain comparable across waves.

**Counter-conditions that would override this bet.** If we observe any of:
- train loss plateau at high residual but held-out SMT-matching still out of reach,
- value-cost correlation staying low even with 10^5+ games per checkpoint,
- MCTS depth refusing to rise above ~5 despite visible policy/value improvement,

then the representation or architecture is the bottleneck after all and we reopen R11's question with harder tests (e.g. graph-structured features, larger task encoders, or non-transformer planners).

# Wave01 Expectations + HPC Config Sanity

**Map:** map_num=2, `5x5_12qb_4gpl_3lyrs`, map_id=`14c99a5d`
**SMT lower bound:** **11** (z3, `baselines/benchmark_smt.py`)
**Do-nothing baseline** (`env.cost_ub`): **24**
**Episode length** (`sum k_t`): **24** steps
**Action space:** 25 (board_size)

## Targets at scale (trained weights)

| Metric                  | Random (wave01)    | Warm-trained goal | Converged goal       |
| ----------------------- | ------------------ | ----------------- | -------------------- |
| `best_cost`             | 21                 | ≤ 14              | **11** (= SMT)       |
| `avg_cost`              | 32.4               | ≤ 18              | ≤ 12                 |
| `avg_cost < cost_ub=24` | false (mean hurts) | true              | true                 |
| `policy_entropy` (mean) | 2.24 (near-flat)   | 1.0–1.8           | 0.3–0.8 (not 0)      |
| `entropy_layer0`        | ≈ log(12)=2.48     | > 1.5 for 80%+ ep | > 0.5 even converged |
| `mcts_depth` (avg)      | 3.17               | 6–10              | 10–20                |
| `mcts_reward_frac`      | 0.84               | > 0.5 ok          | > 0.3 ok             |
| `mcts_boundary_frac`    | 0.46               | 0.4–0.7           | 0.4–0.7              |
| cost spread (p95–p5)    | 9                  | 5–8               | 2–4                  |

**Floor check.** SMT=11, do-nothing=24, random mean=32. Scaled-up selfplay should quickly cross do-nothing (mean < 24), then close on SMT. R05/R10/R12 prior best on map2 = 10–12 at small sim budget; 800 sims should do at least that.

**Anti-collapse.** R13 showed the old `ucb` addend collapsed `entropy_layer0` to 0.65 and under-performed plain `layer_delta`. Scale-up waves must keep `entropy_layer0 > 0.5` at convergence or diversity dies before SMT is reached.

## HPC config review (`neutral_atoms/config_hpc.py` + `config.py`)

Wave01 ran with `preset=hpc` + defaults. Observed on random init:

| Knob                        | Value             | Verdict |
| --------------------------- | ----------------- | --- |
| `mcts.num_simulations`      | 800               | ✓ AlphaDev/AZ parity, `pb_c_base=500` tuned for it |
| `mcts.pb_c_base`            | 500               | ✓ log-term engages meaningfully at N≈500–800 |
| `mcts.pb_c_init`            | 1.25              | ✓ AZ default, sweep later |
| `mcts.root_dirichlet_alpha` | 0.3               | ✓ post-softmax-fix default, sweep 0.1–1.0 next |
| `mcts.root_exploration_fraction` | 0.25         | ✓ AZ default |
| `mcts.known_bounds`         | `{-6, 6}`         | **tight**: wave01 root_value range [-10, -6.2], initial bound clipped. MinMaxStats widens on update so not a correctness bug, but first ~1 sims see unstable normalization. Consider `{-30, 5}` to match `network.value_{min,max}=-20/5`. |
| `mcts.temperature_init`     | 2.0               | ✓ for exploration; at `training_steps=0` always fires |
| `mcts.temperature_decay_steps` | 1000           | Keyed off **`training_steps`** (gradient updates), not selfplay epochs. For fresh-weights data gen (wave01) this is irrelevant. |
| `env.reward_mode`           | `plan_cost`       | R13 finding: `plan_cost` eval_best=10 but self-play entropy collapsed (H=0.34). For scale-up with trained weights, consider `layer_delta + prior_mix_weight=2.0` (R13 prior_mix_w2 matched plan_cost speed with ent preserved). |
| `network.value_min/max`     | `(-20, 5)`        | ✓ covers −cost_ub=−24 tightly (map2). For 8x8 maps, tighten or widen. |
| `network.num_bins`          | 101               | ✓ AZ |
| `network.ema_decay`         | 0.995             | ✓ |
| `mcts.prior_mix_weight`     | 0.0               | off; see reward_mode note |

**What HPC preset leaves out:**
- no `num_selfplay` / `buffer_size` overrides (fine, selfplay_worker ignores these)
- no reward_mode override → inherits `plan_cost` from `config.py`
- no training-side knobs (`batch_size`, `lr`, etc.) → will matter when train_offline.py lands

**Throughput (wave01 observed):**
- 20 nodes × 1000 games / 20 nodes = ~1.0 game/s/node, 62 workers/node
- Per-game: 54s wall × 62 parallel → ~0.87s/sim including env clone overhead
- 20k games in ~17 min wall per node (node wall time 964s)
- Linear: **1 wave = 20k games ≈ 17 min/node × 20 nodes × 62 CPUs**

## Wave01 diagnostics (random-init bootstrap)

- **Cost distribution:** mean 32.4 ± 3.0, median 32, p1=26, min=21 (1 game). **Mean > do-nothing=24**: random policy actively hurts on most games.
- **Weight lineage:** all games from `init` (weight_gen=0). wave01 is strictly a random-policy bootstrap dataset.
- **Per-node variance:** tiny (means 32.23–32.58, depths 3.14–3.24). No node-level bias.
- **Search depth floor:** avg 3.17 across all nodes — priors have no signal so UCB never deepens past first level. Expect depth ~6–10 the moment priors start ranking actions.
- **Signal correlations with cost:** `move_distance` +0.41, `num_moves` +0.31, `noop_frac` −0.31 (intuitive: moving more → more groups → higher cost). `policy_entropy` near zero (random policy, no signal). Training signal is uniform enough — good bootstrap.

## Open questions this wave can't answer

1. Whether trained priors actually close the 32→11 gap at 800 sims (needs weights).
2. Whether `plan_cost` vs `layer_delta+prior_mix` wins at scale (needs paired waves).
3. Whether `dirichlet=0.3` is over/under-exploring at larger branching (needs sweep).

## Next

- Run a wave with a warm-start checkpoint (post-R13 `prior_mix_w2` or R12 fresh spec). Same 20k budget, stamp `weight_gen > 0` and compare distributions.
- If converged cost > 15, tighten `known_bounds` and widen `network.value_min` to −30 for headroom.
- If `entropy_layer0 < 0.5` early, raise `root_dirichlet_alpha` to 0.5 before sweeping.

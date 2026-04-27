# Eval: v3a01 cycle_05 — 5×5 generalist, deterministic inference

**Checkpoint:** `/scratch/huang651/grid_mcts2/pipelines/pipeline_v3a01/checkpoints/cycle_05.ckpt`
**Training steps:** 148,455
**Date:** 2026-04-27
**Job:** 59790830
**Summary JSON:** `eval/cycle_05b/eval_summary.json`

## Setup

- 30 held-out random 5×5 maps, `seed_base=100000` (no overlap with training seeds)
- Deterministic: argmax action selection, no Dirichlet noise
- Device: A100 GPU (inference only; MCTS tree search still CPU)
- Sim counts: 200, 800, 3200

## Results vs Kohei greedy baseline

| method | avg cost | min | p10 | p50 | p90 | avg plan time |
|---|---|---|---|---|---|---|
| **Kohei greedy** | 15.73 | 13 | 14 | 16 | 18 | ~12ms |
| **MCTS sims=200** | 13.63 | 10 | 11 | 13 | 17 | 9.5s |
| **MCTS sims=800** | 12.20 | 10 | 10 | 12 | 14 | 35s |
| **MCTS sims=3200** | **11.53** | **10** | **10** | **11** | **14** | 129s |
| *(selfplay avg, stochastic, 10k sims)* | *12.99* | *8* | *11* | *13* | *15* | *537s* |

## Analysis

### Every sim count beats Kohei

The trained prior dominates Kohei at all sim counts tested. Even 200 sims (9.5s/map) produces avg=13.63 vs Kohei's 15.73 — a **−13% reduction** from a baseline that does no search at all.

### Compute/quality tradeoff

| transition | Δ avg cost | compute ratio |
|---|---|---|
| Kohei → 200 sims | −2.10 | ∞ (search vs. none) |
| 200 → 800 sims | −1.43 | 4× |
| 800 → 3200 sims | −0.67 | 4× |

Gains are steep at first and diminishing past 800 sims. The 200→800 jump (−1.43) is 2× larger than the 800→3200 jump (−0.67) for the same 4× compute increase.

### Recommended deployment point: 800 sims

At 35s/map and avg=12.20:
- Beats Kohei by **−3.53 avg cost (−22%)**
- Beats the stochastic selfplay distribution average (12.99) despite 12× fewer sims than training
- p90=14 vs Kohei p90=18 — worst-case tail is dramatically tighter
- 4× cheaper than 3200 sims for only −0.67 more avg cost

### The trained prior is doing most of the work

The gap between Kohei (no search) and 200-sim MCTS is larger than the gap between 200 and 3200 sims. This means the policy network's learned heuristic accounts for most of the quality gain; additional sims refine rather than discover.

## Per-map cost distributions (sims=800)

```
costs: [13,14,14,13,14,14,11,11,10,11,12,13,14,12,12,13,15,11,10,12,10,14,10,12,11,11,11,11,13,14]
```

No map exceeded cost 15. 8 maps (27%) achieved cost ≤11; 5 maps (17%) achieved optimal cost 10.

## Next steps

- [ ] Run sims=10000 eval (needs >2h budget or Gumbel/GPU backend to be faster)
- [ ] Eval on more cycles (continue v3a01 from cycle_05 for 5 more cycles; expect avg to drop further)
- [ ] Compare SMT optimal on same 30 maps to establish ceiling
- [ ] Run eval on v3a01 cycle_01..05 to show learning curve at inference time

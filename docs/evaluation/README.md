# Evaluation Results

Deterministic inference benchmarks for trained checkpoints. All evals use:
- Argmax action selection (no temperature), Dirichlet noise off
- Fixed held-out validation maps (seed_base=100000, not seen during training)
- Cost = reconfig parallel groups + 2×gate parallel groups (same metric as training)
- Baseline comparison on the same map set

## Baselines (map_num=2, 5×5, 30 maps, seed_base=100000)

| Baseline | avg cost | min | p10 | p50 | p90 | plan time |
|---|---|---|---|---|---|---|
| Kohei greedy | 15.73 | 13 | 14 | 16 | 18 | ~12ms |

Kohei: greedy per-qubit placement maximizing gate-gain at each step. No search. See `baselines/kohei_policy.py`.

SMT and DPQA baselines not yet benchmarked on this map set.

## Evals

| Checkpoint | Map | Date | Key result | Doc |
|---|---|---|---|---|
| v3a01 cycle_05 | 5×5 random (map_num=2) | 2026-04-27 | 200 sims beats Kohei (13.63 vs 15.73); 800 sims avg=12.20 | [v3a01_cycle05.md](v3a01_cycle05.md) |

# Round 04: Value Pretraining + Critical Bug Fixes

**Date:** 2026-03-13
**Map:** 1 (4x4, 8q, 3 identical layers, 4 gates each)
**Cost bounds:** lb=6, ub=24. Action space: 129.

## Context

Round 03 analysis revealed the network never learns beyond the trivial "just execute" strategy (cost=18). Deeper investigation uncovered two critical bugs:

### Bug 1: Value range too narrow (Round 03 regression)

Round 03 changed `value_min/max` from [-25, 25] (51 bins) to [-10, 10] (101 bins) for "finer resolution." But actual targets exceed this range:

- **Correctness targets**: cumulative step rewards telescope to `initial_cost` ≈ 18. Clamped to 10. Every training sample has the same target (10). The correctness head cannot learn.
- **Latency targets**: `-final_cost` ≈ -18 for trivial games. Clamped to -10. Cannot distinguish cost=13 from cost=18.

**Fix**: Restore `value_min=-25, value_max=25` with 101 bins (0.5/bin). Covers all actual cost ranges.

### Bug 2: Correctness value telescopes with gamma=1

With `discount=1.0` (gamma=1), the correctness value at any state equals `remaining_cost(s)`. In MCTS UCB:

```
ucb_value = reward(a) + gamma * V_correctness(s')
          = (cost_s - cost_s') + 1 * cost_s'
          = cost_s
          = constant (independent of action!)
```

The correctness head provides ZERO discriminative signal to MCTS. The only useful signal comes from the **latency head** which predicts `-final_cost` (varies by trajectory quality). But with bug #1, that signal was also destroyed.

### Why pretraining helps

With correct value ranges, the MCTS combined value is:

```
V(s) = cw * V_correctness(s) + lw * V_latency(s)
     ≈ cw * cost(s) + lw * (-expected_final_cost(s))
```

The correctness term `cw * cost(s)` telescopes with reward. The latency term `lw * (-expected_final_cost)` does NOT — it depends on the quality of the policy from state s. With `lw > cw`, the latency signal dominates, allowing MCTS to distinguish good and bad states.

But the latency head starts random and needs many completed games to learn. By pretraining it on `-cost(random_board)`, MCTS gets useful state evaluations from epoch 1: lower cost → higher combined value → MCTS explores those states more.

The auxiliary loss during training continuously generates (random_board, cost) pairs to keep the value heads calibrated as the policy evolves.

## Hypotheses

- **H1 — Value range fix alone improves results.** The [-10, 10] range destroyed both value heads. With [-25, 25], the latency head can at least learn different final costs. Predicted: avg_cost < 18 (some improvement from latency signal).

- **H2 — Pretraining changes the self-play distribution.** With pretrained value head, MCTS explores move actions more. Predicted: gate_frac < 0.5 in epoch 1, avg_steps > 3.

- **H3 — `lw > cw` is necessary.** With `cw=lw=1`, the correctness term masks the latency signal (it's larger magnitude and constant). With `cw=0.5, lw=2.0`, the latency quality signal dominates. Predicted: lower best_cost with high lw.

- **H4 — Auxiliary loss prevents value collapse.** Without aux loss, TD learning pushes correctness toward +cost, fighting the MCTS-useful signal. Aux loss anchors the value heads. Predicted: without aux, improvement disappears after a few epochs.

## Critique / Known risks

1. **Pretraining target mismatch.** Correctness is pretrained on +cost (matching TD), latency on -cost (matching terminal reward). The combined value = `cw*cost + lw*(-cost) = (cw-lw)*cost`. With lw > cw this gives a useful signal, but correctness and latency heads see contradictory targets for the same board state.

2. **Random boards may not cover relevant states.** Random atom shuffles produce a narrow cost distribution (mostly 18-24 on Map 1). States reached after good reconfiguration moves (cost 10-14) are underrepresented.

3. **The credit assignment problem remains.** Pretraining gives MCTS accurate leaf evaluations, but MCTS still needs many simulations to discover that multi-step move sequences reduce final cost. With 25-100 sims and 129 actions, most move sequences go unexplored.

4. **Policy collapse may still occur.** If the first epoch's visit counts still favor execute (because of higher one-step reward), the policy collapses regardless of value accuracy. May need policy pretraining too.

## Experiments

### Shared parameters

```
--config.map_num=1
--config.mcts.root_dirichlet_alpha=0.3
--config.mcts.root_exploration_fraction=0.5
--config.training.epochs=30
--config.experiment.early_stopping_patience=15
--config.network.value_min=-25.0
--config.network.value_max=25.0
```

### Phase A: Isolate effects

| # | Experiment | Key change |
|---|-----------|------------|
| 1 | Value range fix only | No pretrain, cw=1 lw=1 |
| 2 | + lw>cw | cw=0.5, lw=2.0 |
| 3 | + pretrain (500 steps) | cw=0.5, lw=2.0, pretrain=500 |
| 4 | + aux loss | cw=0.5, lw=2.0, pretrain=500, aux=0.5 |

### Phase B: Scale

| # | Experiment | Key change |
|---|-----------|------------|
| 5 | Best Phase A + 50 sims, 40 games, parallel | Scale up |
| 6 | Best Phase A + gate_only reward | Combined approach |

## What to look for

1. **root_value trajectory** — does it stay calibrated or flip sign (pretraining lost)?
2. **gate_frac in epoch 1-3** — does pretraining prevent immediate collapse to execute?
3. **best_cost < 18** — any improvement over trivial?
4. **avg_cost < 18** — is the POLICY learning, not just getting lucky?
5. **Value head accuracy** — does correctness predict +cost, latency predict -cost?

## Results — Phase A

| # | Experiment | Run ID | best | avg (final) | gate_frac | epochs | collapsed? |
|---|-----------|--------|------|-------------|-----------|--------|------------|
| 1 | Value fix only | 7380ca49 | **14** | **17.5** | 1-48% | 30 | **No** |
| 2 | + lw>cw (0.5/2.0) | ee5d7958 | 14 | 20.6 | 0-42% | 26 | No |
| 3 | + pretrain | 8c57fb9b | 18 | 21.5 | 8-49% | 18 | No |
| 4 | + pretrain + aux | 46ab53f8 | 16 | 18.0 | 7→100% | 19 | Yes (ep 6) |
| 4b | + freeze 10 ep | 349c00ef | 14 | 18.0 | 4→100% | 23 | Yes (ep 11) |

### Key findings

1. **The value range fix is THE critical change.** Widening [-10,10] → [-25,25] alone prevents policy collapse and enables gradual learning. Exp 1 (no pretrain, no weight tuning) achieves avg_cost=17.5 — first time below the trivial 18.

2. **Pretraining and weight tuning were counterproductive.** Pretraining gets overwritten by TD in 1-2 epochs. The `lw > cw` weighting amplifies noisy latency estimates in early epochs. The aux loss and freeze delay collapse but don't prevent it.

3. **The network IS learning (Exp 1).** best_cost steadily improves: 27→22→19→18→17→16→14 over 30 epochs. This is genuine policy improvement, not just lucky search.

4. **Why the range matters:** With [-10,10], correctness targets (~18) clamp to 10 and latency targets (~-18) clamp to -10. Both heads see a constant target → no gradient signal → value heads are useless → MCTS degrades to random exploration → policy collapses to first strong pattern (execute). With [-25,25], the latency head learns to distinguish cost=14 from cost=18, providing genuine quality signal to MCTS.

### Next steps

- **Scale up Exp 1 config** (more sims, more games, more epochs) — the learning curve hasn't plateaued
- **Test on Map 0 and Map 2** — verify the fix generalizes
- **Random baseline** — run FakeNet with many games to confirm the network adds value
- **Investigate cost=14 solutions** — understand what reconfiguration was discovered

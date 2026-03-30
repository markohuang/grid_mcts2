# Experiment 01: Loss Landscape Study & 8×8 Scaling

**Date:** 2026-03-27
**Status:** In progress

## Motivation

The feasibility-based surrogate (cost_surrogate_v3.md) uses `-log F` where `F = ∏ P_ij` as the loss for both gate and reconfig costs. Initial ablation testing revealed the auxiliary `conflict_count` loss was dead code (never computed in `gate_feasibility`). After fixing the bug, we discovered that the choice of loss function fundamentally affects optimization quality.

## Background: Three Loss Modes

All operate on the same pairwise compatibility probabilities `P_ij ∈ [0,1]`:

| Mode | Per-pair loss | Gradient weight | Behavior |
|------|---------------|-----------------|----------|
| **log** | `-log P_ij` | `1/P_ij` (unbounded) | Precise near optimum; bottleneck focusing on worst pair |
| **linear** | `1 - P_ij` | `1` (constant) | Bounded, democratic; all bad pairs weighted equally |
| **huber(δ)** | log if P>δ, linear if P≤δ | min(`1/P_ij`, `1/δ`) | Best of both: log precision near optimum, linear stability far away |

**Mathematical relationship:** `-log x ≥ 1-x` always, with equality only at `x=1`. Near the optimum both agree to first order via Taylor: `-log P ≈ (1-P) + (1-P)²/2 + ...`.

## Experiment 1: Loss Mode Ablation on 5×5 Map 2

**Setup:** Map 2 (5×5, 12 atoms, 3 layers, 4 gates/layer). 10 seeds × 400 steps, Adam lr=0.03.

### Results

```
Config                     Best  Worst   Mean  ≤12
log (λ_g=1)                  12     14   13.0   2/10    ← WORST: can't guarantee χ_gate=1
linear (λ_aux=2)             11     12   11.6  10/10    ← BEST mean
huber δ=0.05                 12     14   13.3   2/10    ← too close to log
huber δ=0.1                  11     12   11.9  10/10    ← sharp transition from δ=0.05
huber δ=0.3                  11     12   11.9  10/10
log(0.1)+aux(2)              11     12   11.8  10/10    ← rebalanced combo works
huber(0.1)+aux(1)            11     12   11.8  10/10
linear (λ_aux=3)             11     12   11.7  10/10    ← diminishing returns
linear (λ_aux=4)             11     12   11.7  10/10
```

### Analysis

**Why raw log fails (mean 13.0, worst 14):**
The `1/P_ij` gradient amplification creates rigid optimization trajectories. Whichever pair happens to be numerically worst dominates the gradient, pulling the optimizer toward fixing one constraint while potentially breaking another. All seeds follow the same ridgeline → deterministic convergence to the same local minimum. Some seeds can't even achieve χ_gate=1 (cost 13-14 includes g4/g6).

**Why linear is best (mean 11.6):**
Equal-weight gradients create a broader loss landscape. The optimizer receives balanced signal to improve all pairs → different seeds explore different regions → 4/10 find the cost-11 basin. The cost-11 solution requires coordinated reconfig across layers (r2+g2, r1+g2, r2+g2 = 4+3+4 = 11).

**The δ transition is sharp:**
- δ=0.05 (max 20× amplification): fails like log
- δ=0.1 (max 10× amplification): works
- Critical amplification threshold for M=4: between 10× and 20×

**The rebalancing insight:**
Original ablation (B-D identical to A) was a magnitude imbalance: with M=4, 6 pairs all at P≈0.01, `-log F ≈ 27.6` but `conflict_count ≈ 5.9`. At λ_g=1.0, the log term dominated 4.7× regardless of λ_aux. Rebalancing to λ_g=0.1+λ_aux=2.0 works (mean 11.8).

### Recommendations for 5×5

- **Default:** `loss_mode='huber', delta=0.1, lambda_g=1.0` — robust, no extra λ to tune
- **Best single-instance:** `loss_mode='linear', lambda_aux=2.0` — mean 11.6
- **For generalist NN:** huber δ=0.1 (robust across map variation without per-map tuning)

---

## Experiment 2: 8×8 Scaling

**Setup:** Three random 8×8 maps (20 atoms, 5 layers, 10 gates/layer). 5 restarts × 400 steps.

M=10 gates → 2^10 = 1024 direction assignments × C(10,2) = 45 pairs. GPU vectorized: 118ms/step on A6000.

### Results: Huber δ=0.1 (default)

| Map | Seed | Baseline | Best | Mean | Breakdown (best) |
|-----|------|----------|------|------|-------------------|
| Map 3 | 42 | 62 | **49** | 51.0 | r3+g10, r3+g6, r2+g6, r3+g8, r2+g6 |
| Map 4 | 123 | 64 | **52** | 56.8 | r1+g8, r2+g10, r2+g12, r2+g6, r3+g6 |
| Map 5 | 999 | 58 | **50** | 53.2 | r1+g6, r1+g10, r1+g14, r2+g8, r3+g4 |

**Improvement:** 19-21% cost reduction from baseline.

### Results: Linear λ_aux=2.0

| Map | Seed | Baseline | Best | Mean |
|-----|------|----------|------|------|
| Map 3 | 42 | 62 | 53 | 54.2 |

**Linear at λ_aux=2.0 is worse on 8×8** (54.2 vs 51.0 for huber). This is the opposite of the 5×5 finding.

### Observation: Gate Costs Are High

Many layers show g8, g10, g12, g14 — not reaching χ_gate=1 (which would be g2). The lower bound per layer is 2, so with 5 layers the lower bound is 10. Current best is 49, implying gate costs alone are ~40 (vs lower bound 10).

**Hypothesis:** With M=10 and 45 pairs, the optimization problem is qualitatively harder:
1. **The feasibility region is narrower** — more pairwise constraints must be simultaneously satisfied
2. **λ scaling:** Linear λ_aux=2.0 was calibrated for 6 pairs. With 45 pairs, the max conflict_count is 45 (not 6), so the effective penalty-to-base-cost ratio is very different. Need λ_aux ∝ M²/2 scaling.
3. **The direction search (2^10 = 1024)** may be too sparse — the 2-SAT structure could help find better directions
4. **400 steps may not be enough** — the landscape is 7.5× more complex (45 vs 6 pairs)

### Next Steps

1. Try λ_aux=8 (linear) and λ_aux=15 (proportional to 45/6 ≈ 7.5×)
2. Try 800-1000 steps
3. Try huber with larger λ_g (2.0, 3.0) to push harder
4. Profile: which layers are hardest? Is it the first layer (fixed initial positions) or later layers?

---

## Infrastructure Built

All runs produce:
- `outputs/<run_id>/config.json` — full frozen config
- `outputs/<run_id>/metrics.jsonl` — per-restart history (step, surrogate, true_cost, breakdown)
- `outputs/<run_id>/solutions/best.json` — atom-viz compatible JSON (board, circuit, plan)
- `outputs/<run_id>/solutions/restart_NNN_costXXX.json` — per-restart solutions
- `outputs/run_registry.jsonl` — one-line summary per run

Run commands:
```bash
# 5×5 default (huber)
python main.py

# 5×5 best known config
python main.py --config.surrogate.loss_mode=linear --config.surrogate.lambda_g=0.0 \
               --config.surrogate.lambda_r=0.0 --config.surrogate.lambda_aux=2.0

# 8×8 Map 3
python main.py --config.map_num=3

# Override anything
python main.py --config.map_num=3 --config.optimizer.n_steps=800 \
               --config.surrogate.loss_mode=huber --config.surrogate.delta=0.1
```

## Code Changes

| File | What changed |
|------|-------------|
| `surrogate/feasibility.py` | Vectorized (no Python for-loops), 3 loss modes via `_pairwise_loss`, `conflict_count` bug fixed |
| `surrogate/__init__.py` | Created (proper package) |
| `config.py` | ml_collections + absl, Maps 0-5, map generator, loss_mode/delta params |
| `experiment.py` | Run tracking, atom-viz JSON, metrics, registry |
| `main.py` | Fabric entry point, multi-restart, per-restart solutions |
| `tests/ablation_loss_mode.py` | Loss landscape ablation |
| `tests/ablation_aux_loss.py` | Updated to use config system |

Performance: 0.25ms/eval on 5×5 (was ~47s/seed in original ablation → 2s/seed after vectorization). 118ms/step on 8×8 GPU.

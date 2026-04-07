# Experiment 04: Partition Function Chromatic Surrogate

## Date
2026-04-06

## Motivation

The Erdős/Potts surrogate (`erdos.py`) has a fundamental gradient pathology: auxiliary coloring variables (S) converge quickly and then **screen placements from the loss**. At equilibrium, both the Potts energy (∂/∂A = 0 because Potts=0) and the group count (∂/∂placement = 0 because it depends only on S) have zero gradient w.r.t. placements. The optimizer cannot discover that moving atoms would reduce the chromatic number.

The feasibility surrogate (`feasibility.py`) avoids auxiliary variables but counts pairwise conflicts, not the chromatic number. An SMT-optimal solution with many atom moves but low χ_G is penalized by feasibility despite being low-cost.

## Approach

Use the **partition function** (soft chromatic polynomial) as a differentiable proxy for χ:

```
Z̃(A, q) = Σ_{σ ∈ [q]^M} ∏_{i<j : σ_i=σ_j} (1 - A_ij)
```

At discrete A ∈ {0,1}, this equals the chromatic polynomial P(G,q). At soft A, it smoothly interpolates. **No auxiliary variables** — Z̃ is a pure polynomial in the conflict matrix entries.

Two aggregation modes tested:

**Sigmoid mode** (χ̂ estimator):
```
χ̂(A) = 1 + Σ_{q=1}^{q_max} σ(-α · log Z̃(A, q))
```

**Cost mode** (non-saturating):
```
cost(A) = -Σ_{q=1}^{q_max} softplus(log Z̃(A, q)) / q²
```

Unified layer cost: `C(t) = p_any_move · f(A_R) + 2 · f(A_G)` where f is either mode. Both A_R and A_G depend on `curr_dists`, so gradients capture the reconfig/gate tradeoff.

## Complexity

Z̃ requires enumerating q^M colorings. For the target problem sizes:
- Gate (M_G ≤ 5): 5^5 = 3,125 — trivial
- Reconfig (M_R ≤ 10): 5^10 ≈ 10^7 — feasible

Compared to graph enumeration (2^{M(M-1)/2}), the partition function scales as q^M — exponential in nodes rather than edges. For M=10 this is 10^7 vs 10^13.

## Perfect Graph Analysis

Before evaluating, we checked whether AOD conflict graphs are perfect (which would make Lovász θ exact, an alternative approach). Results from 500 random trials per config:

| Graph type | M | 5×5 grid | 8×8 grid |
|---|---|---|---|
| Gate | 3-4 | 100% perfect | 100% perfect |
| Gate | 5 | 100% | 99.8% |
| Reconfig | 4-6 | 100% | 100% |
| Reconfig | 8 | 99.2% | 98.0% |
| Reconfig | 10 | 96.4% | 93.4% |

Conclusion: gate graphs are essentially always perfect. Reconfig graphs develop odd C₅ holes at M≥8. The partition function approach avoids this limitation entirely.

## Results

### Test 1: Discrete Recovery

χ̂ (α=20) matches true χ for all tested graphs: empty (1), single edge (2), K₃ (3), K₄ (4), C₅ (3), K_{2,3} (2). The erdos.py surrogate incorrectly estimated 4 gate groups per layer when the true χ_G was 3.

### Test 2: Gradient Optimization

Setup: Map 2 (5×5, 12 atoms, 3 layers × 4 gates). Do-nothing baseline = 18. Same initialization as erdos.py test (300 Adam steps, lr=0.05).

#### Sigmoid mode — α sensitivity

| α | True cost | Breakdown | Atoms moved | Notes |
|---|---|---|---|---|
| 10.0 | 18 | r0+g6 × 3 | 0 | Sigmoid saturated, ~zero gradients |
| 3.0 | 17 | r0+g6, r0+g6, r3+g2 | 3 | Layer 2 reduces to χ_G=1 |
| 1.0 | **14** | r0+g6, r2+g2, r2+g2 | 4 | Layers 1-2 achieve χ_G=1 |
| 0.5 | **14** | r2+g4, r0+g4, r2+g2 | 5 | Different configuration, same cost |

**Key finding**: α=10 (default) causes complete sigmoid saturation. Lowering α to 1.0 enables gradient flow, reducing true cost from 18 → 14 (22% improvement).

#### Cost mode (-softplus/q² formulation)

| Init bias | LR | True cost | Breakdown | Atoms moved |
|---|---|---|---|---|
| 4.0 | 0.05 | **11** | r2+g2, r1+g2, r2+g2 | 6 |
| 4.0 | 0.20 | **12** | r2+g2, r2+g2, r2+g2 | 6 |
| 2.0 | 0.05 | 13 | r1+g4, r2+g2, r2+g2 | 6 |
| 2.0 | 0.20 | 13 | r1+g4, r2+g2, r2+g2 | 8 |
| 1.0 | 0.05 | 15 | r1+g6, r1+g4, r1+g2 | 6 |
| 0.5 | 0.05 | 16 | r2+g6, r1+g2, r1+g4 | 9 |

**Key finding**: cost mode with bias=4.0 achieves true cost **11** (39% improvement from 18). It consistently reduces gate groups to χ_G=1 across layers, accepting 1-2 reconfig groups as the tradeoff. Gradient norms are 2-3× higher than sigmoid mode.

Paradoxically, **higher initial bias (more peaked distributions) works better** with cost mode. Lower bias (softer distributions) leads to worse results — the optimization starts further from any good discrete solution and the landscape becomes more complex.

### Comparison Summary

| Approach | True cost | Improvement | Gradient norm | Aux vars? |
|---|---|---|---|---|
| Erdős/Potts | 18 | 0% | ~0 for placements | Yes (coloring S) |
| Feasibility | (not directly comparable) | — | Strong but misaligned | No |
| **Partition sigmoid α=10** | 18 | 0% | ~0 (saturated) | No |
| **Partition sigmoid α=1** | 14 | 22% | 0.009 | No |
| **Partition cost mode** | **11** | **39%** | 0.021 | No |

### Speed

- Forward: 45ms per evaluation (3 layers, M_G=4, M_R=8)
- Forward + backward: 140ms
- Acceptable for optimization loops. Would need GPU batching for training integration.

## Analysis

### Why sigmoid saturates

At near-one-hot placements, Z̃(q) is either ≈0 (q < χ) or a large integer (q ≥ χ). With α=10: σ(-10 × log 6) = σ(-18) ≈ 0, and σ' ≈ 0. The sigmoid is deep in its saturated regime, cutting off all gradient flow to placements.

### Why cost mode works

`-softplus(log Z̃(q))/q²` avoids the sigmoid entirely. Its gradient is:

```
∂cost/∂A_ij = -Σ_q (1/q²) · σ(log Z̃(q)) · (1/Z̃(q)) · ∂Z̃(q)/∂A_ij
```

The factor σ(log Z̃) ∈ (0,1) is non-zero whenever Z̃ > 0 (always true for soft A). The 1/q² weighting prevents domination by large-q terms where Z̃ is exponentially large. This gives useful gradients at **any** operating point.

### The 1/q² weighting

Without weighting, large q terms dominate (more colorings → larger Z̃ → larger softplus). The 1/q² down-weights them, focusing optimization on the critical transition q ≈ χ. Alternative weightings (1/q, 1/q³, exponential decay) may also work — not explored.

### Reconfig-gate tradeoff

The cost mode correctly navigates the tradeoff: moving atoms to positions where gates become compatible (χ_G: 3→1) while accepting small reconfig costs (χ_R: 0→1-2). The unified formulation `p_any · cost_R + 2 · cost_G` lets gradient descent find this balance automatically.

## Ranking Validation

Tested whether the surrogate correctly ranks discrete solutions by true cost. Generated 21 solutions per instance (do-nothing + random perturbations with 1-6 atom moves) across 16 instances (2 seeds × {3,4} gates × {3,4} layers × {5×5, 8×8}).

### Calibration at one-hot

With direction enumeration (best of 2^M_G directions), sigmoid χ̂ matches true cost within ±1 for 8/10 test solutions on Map 2. The small mismatches come from the greedy coloring heuristic in `true_cost` not always finding the optimal χ.

The gate 2× weighting is correctly reflected:
- True cost = Σ_t [χ_R(t) + 2·χ_G(t)]
- Sigmoid cost = Σ_t [p_any·χ̂_R + 2·χ̂_G]
- At one-hot: these match (p_any = 𝟙[move], χ̂ = χ)

Reconfig χ̂_R shows consistent +0.5 bias from the empty-graph sigmoid artifact (Z̃(1)=1, σ(0)=0.5). This is a constant offset that doesn't affect ranking.

### Ranking correlations (Spearman ρ)

| Variant | Mean ρ | Min ρ | Max ρ | N |
|---|---|---|---|---|
| **sigmoid (one-hot)** | **0.946** | 0.833 | 0.976 | 16 |
| cost (one-hot) | 0.709 | 0.541 | 0.847 | 16 |
| sigmoid (noisy) | 0.851 | 0.644 | 0.942 | 16 |
| **cost (noisy)** | **0.939** | 0.849 | 0.976 | 16 |

Key findings:
1. **Sigmoid mode ranks best at one-hot** (ρ=0.946) because it's calibrated — values match true cost.
2. **Cost mode ranks poorly at one-hot** (ρ=0.709) because `-softplus/q²` is not calibrated to group counts.
3. **Cost mode improves dramatically with noise** (ρ=0.939) — the non-saturating gradients help when distributions are soft.
4. **Sigmoid degrades with noise** (ρ=0.851) — sigmoid saturation at near-one-hot distributions introduces ranking errors.

This reveals a fundamental tradeoff:
- **Sigmoid**: accurate ranking at discrete solutions, poor gradients for optimization
- **Cost**: good gradients for optimization, poor ranking at discrete solutions

### Pairwise violations

| Variant | Violations | Total pairs | Rate |
|---|---|---|---|
| sigmoid | 239 | 3234 | 7.4% |
| cost | 517 | 3234 | 16.0% |

### Implication for MCTS integration

For MCTS value targets (where we evaluate discrete board states), sigmoid mode is the right choice — it's calibrated and ranks well. For direct gradient optimization of placements, cost mode is better. An annealing strategy (cost mode early → sigmoid mode late) could combine both benefits.

## Limitations

1. **Not calibrated**: cost mode loss values don't correspond to true cost (different units). For MCTS value targets, would need the sigmoid mode with careful α tuning, or a separate calibration step.
2. **Local optima**: different initializations find different solutions (11 vs 12 vs 13). The loss landscape has multiple basins.
3. **Scaling**: q^M enumeration limits to M ≤ 10-12 on CPU. For larger circuits, would need GPU parallelism or Monte Carlo Z̃ estimation.
4. **Collision avoidance**: current formulation doesn't prevent two atoms from occupying the same cell. Would need an additional penalty or constrained optimization.

## Files

- `surrogate/partition.py` — Implementation (log_Z_tilde, chromatic_estimate, chromatic_cost, PartitionSurrogate)
- `tests/test_partition.py` — Full test suite with ablations
- `tests/test_perfect_graph.py` — AOD conflict graph perfectness analysis

## Next Steps

1. Add collision penalty to prevent atom overlaps in optimized solutions
2. Explore α annealing: start with low α (strong gradients) → increase α (accurate χ)
3. Integrate with MCTS: use partition surrogate as value function target or direct search heuristic
4. Test on 8×8 maps with 5 gates per layer (larger M_R, M_G)
5. Investigate the connection between cost mode and actual χ for calibration

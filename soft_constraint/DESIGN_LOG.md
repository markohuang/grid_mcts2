# Design Log — Differentiable Cost Surrogate

Chronological record of design decisions, experiments, and rationale.

---

## Session 1: Problem Formulation

**Goal:** Build a differentiable surrogate cost that operates on soft placement distributions (logits) and outputs an unbiased estimator monotonically correlated with the true parallel-move cost.

**Priorities (ranked):** Unbiasedness > Low variance > Speed > Tightness.

**Key structural insight:** Gates within a layer form a matching (no shared atoms). This means gate states are independent — a critical property exploited by the enumeration approach.

---

## Decision 1: Edge-Independent Enumeration (v1) — ABANDONED

**Approach:** Compute marginal P(conflict) per gate pair, assume edges independent, enumerate all 2^E conflict graphs weighted by P(edge config), look up χ.

**Result:** Significant bias (+0.23 to +0.36) and catastrophic gradient misalignment (cosine similarity 0.11). The bias came from correlated edges (edges sharing a gate depend on the same atoms' positions).

**Lesson:** Forward-pass bias and gradient bias are different problems. Moderate value bias (5%) caused near-total gradient decorrelation. The independent-edge factorization creates artificial gradient pathways.

**Decision:** Abandoned as primary surrogate. The edge-independence assumption is fundamentally flawed for this graph structure.

---

## Decision 2: Exact Gate-State Enumeration (v2) — ADOPTED

**Approach:** Enumerate over the joint gate-state space (D^M where D = C² per gate). Since gates are a matching, gate states are independent → the joint probability factorizes. For each joint state, look up the 6-bit conflict graph and retrieve χ from a precomputed table.

**Result:** Zero bias (confirmed within MC sampling error). Gradient cosine similarity 1.000000 against numerical FD. But O(D^4) is too slow at full enumeration (~250s on CPU for soft distributions).

**Decision:** Adopted as the theoretically correct approach. Speed optimization needed.

---

## Decision 3: Top-K Pruning + GPU Vectorization (v3) — ADOPTED

**Approach:** Keep only top_k cells per atom, giving K = top_k² gate states per gate. Fully vectorize the (K0, K1, K2, K3) computation as a single GPU tensor operation. Renormalize pruned distributions.

**Result:** 67ms at top_k=8 with canonicalization on GPU. Gradient still 1.000000. Pruning bias ~-0.4 at σ=4.0 (vanishes as distributions sharpen).

**Decision:** Adopted. The pruning bias is acceptable because (a) it's conservative (underestimates cost), (b) it vanishes at convergence, and (c) the gradient direction is preserved.

---

## Decision 4: REINFORCE Baseline — INFORMATIVE FAILURE

**Finding:** The cost landscape is nearly flat along individual logit directions. Perturbing a single atom's logit by ±0.5 changes expected cost by ~0.03, while MC standard deviation is ~0.37. MC finite-difference "ground truth" is unreliable (two independent FD estimates have cosine similarity 0.08).

**Lesson:** This is the coordination problem described in the problem doc. Individual atom moves barely affect cost; only coordinated multi-atom moves matter. MC-based methods (REINFORCE, FD) cannot resolve the tiny gradient signals. The exact analytical surrogate is essential.

**Decision:** MC approaches abandoned for gradient computation. The exact surrogate is the only viable path for this problem structure.

---

## Decision 5: Inversion-Based Reconfig Surrogate — ADOPTED

**Insight:** AOD compatibility = coordinate-wise monotonicity. The column constraint says source column ordering must equal destination column ordering. Same for rows. This is two independent 1D monotonicity conditions.

**Approach:** Count expected pairwise ordering inversions on each axis. O(N² V⁴) where V = max(H,W). No top-K needed.

**Key property:** I=0 ⟺ χ=1 (exact at the optimum). Pearson correlation with true χ: 0.92. Calibrated: χ ≈ 1.96 + 0.244 × max(I_col, I_row).

**Limitation:** Axis decomposition gap of 24% — destination collisions not captured by per-axis analysis. Mitigated by collision penalty term.

**Decision:** Adopted. The I=0 ⟺ χ=1 property is the critical one since good solutions have χ_reconfig=1.

---

## Decision 6: Interpretation A (All Layers at Once) — ADOPTED

**Options considered:**
- A: Model outputs all layers' distributions simultaneously. Independent per-layer gate costs, coupled reconfig costs between consecutive layers.
- B: Sequential/autoregressive. Each layer conditioned on previous outcomes.

**Decision:** Interpretation A. Simpler (no sequential sampling, no distribution propagation), fully differentiable, and sufficient for the planning problem where we know the full task sequence upfront.

---

## Decision 7: Collision Handling — POST-PROCESSING

**Options considered:** Soft penalty, Sinkhorn projection, autoregressive masking, post-processing.

**Decision:** Post-processing via greedy confidence-based assignment. Rationale: collisions are rare and localized; the optimizer finds the right neighborhood, post-processing snaps to a valid solution. Track degradation to validate this assumption.

---

## Decision 8: Reconfig Scope — ALL ATOMS

**Issue:** Initial implementation only computed reconfig cost for relevant atoms (those in the current layer's gates). This missed the cost of non-relevant atoms that moved.

**Fix:** Reconfig cost should consider all atoms. If an atom doesn't have free logits (not relevant), it stays put and contributes zero reconfig cost naturally. But if any atom moves (relevant or not), it should be counted.

---

## Open Decision: Gate Cost as Feasibility Check

**Observation:** Since good solutions always achieve χ_gate = 1, the full χ computation may be overkill. Instead of computing E[χ], we could check: "are all gate moves pairwise compatible?" This is O(M² V⁴) — much cheaper than the full O(K^(2M)) enumeration.

**Potential approach:** Gate cost = 2 + penalty × P(any incompatibility). When P=0, cost is exactly 2 (optimal). When P>0, cost increases.

**Status:** Implemented as `surrogate/feasibility.py`. Under testing.

---

## Decision 9: Unified Feasibility Framework — ADOPTED

**Insight:** Both gate cost and reconfig cost are instances of the same mathematical object: checking coordinate-wise order preservation of move vectors. The AOD constraint = monotonicity on each axis independently.

**Gate cost:** $2 + \lambda_g \cdot (-\log \hat{F}_{\text{gate}})$ where $\hat{F} = \max_{\mathbf{d}} \prod P(\text{compat}_{ij} \mid \mathbf{d})$
**Reconfig cost:** $P(\text{any move}) \cdot (1 + \lambda_r \cdot (-\log \hat{F}_{\text{reconfig}}))$ where $\hat{F} = \prod P_{\text{eff}}(\text{compat}_{ij})$

Both use $O(V^4)$ pairwise computations from coordinate marginals. No top-K pruning needed.

**Key properties:**
- Exact at optimum: $\hat{F} = 1 \iff \chi = 1$ (proven via forward/backward implication)
- Conservative bias: FKG inequality guarantees edge-independent product ≤ true joint probability
- Bottleneck-focusing gradients: $-\log \prod P_{ij} = \sum -\log P_{ij}$ concentrates gradient on worst pair
- Polynomial complexity: $O(2^M \cdot M^2 \cdot V^4)$ for gates (2-SAT for larger M), $O(N^2 \cdot V^4)$ for reconfig

**Replaces:** Exact χ enumeration (gate_cost.py) for the primary optimization. Exact enumeration retained for validation/benchmarking.

**Theoretical connections:** 2-SAT structure of direction constraints, FKG inequality for bias direction, Knothe-Rosenblatt rearrangement for feasibility existence, Dilworth's theorem for axis decomposition.

---

## Decision 10: Reconfig Includes All Atoms — ADOPTED

**Issue:** Previous implementation only computed reconfig cost for relevant atoms (those in current layer's gates). This ignored the cost of non-relevant atoms that moved.

**Fix:** `mover_indices` defaults to all atoms. The `P(both move)` weighting naturally zeros out the contribution of atoms that stay, so including non-movers has zero computational cost and ensures correctness.

**Design:** For each atom pair, the effective compatibility probability is:
$P_{\text{eff}} = (1 - P(\text{both move})) + P(\text{both move}) \cdot P(\text{compat})$
This interpolates between 1 (if neither moves → no conflict) and $P(\text{compat})$ (if both move → check compatibility).
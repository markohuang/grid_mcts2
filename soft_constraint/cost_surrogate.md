# Differentiable Cost Surrogate for Neutral Atom Reconfiguration

## 1. Problem Formulation

### 1.1 The Hardware Problem

A quantum circuit is executed on a grid of optical tweezers (H × W) holding neutral atoms (qubits). Two-qubit gates require physically moving atom pairs together via AOD lasers, then separating them. AOD lasers move entire rows/columns simultaneously, creating geometric constraints on which moves can execute in parallel.

The circuit is decomposed into **layers**, each containing a set of two-qubit gate pairs (a matching — no atom appears twice per layer). Between layers, atoms can be **reconfigured** (moved to new positions) to improve the geometry for upcoming gates.

### 1.2 Cost Structure

For each layer $t$:

$$\text{layer\_cost}(t) = G_t^{\text{reconfig}} + 2 \cdot \chi_t$$

where:
- $G_t^{\text{reconfig}}$ = number of sequential AOD groups needed to move atoms from previous positions to new positions (only atoms that actually move contribute; no-ops are free)
- $\chi_t$ = chromatic number of the gate conflict graph at the final positions (minimum parallel groups to execute all gates)
- The factor of 2 accounts for the round-trip: move atoms together (enter) + separate them (exit)

**Total cost:**
$$C = \sum_{t=0}^{T-1} \left( G_t^{\text{reconfig}} + 2 \cdot \chi_t \right)$$

### 1.3 The AOD Parallel Compatibility Constraint

Two moves $i, j$ with source/destination positions can execute in the same parallel group iff:
- **Row preservation:** either both share source/dest rows, or neither shares and the relative order is preserved
- **Column preservation:** same logic on the column axis
- **No collision:** different destinations

Formally, for displacement differences $(\Delta c^s, \Delta c^d)$ in columns and $(\Delta r^s, \Delta r^d)$ in rows:

$$h\_ok = (\Delta c^s = 0 \wedge \Delta c^d = 0) \;\lor\; (\Delta c^s \neq 0 \wedge \Delta c^d \neq 0 \wedge \text{sign}(\Delta c^s) = \text{sign}(\Delta c^d))$$

$$v\_ok = (\Delta r^s = 0 \wedge \Delta r^d = 0) \;\lor\; (\Delta r^s \neq 0 \wedge \Delta r^d \neq 0 \wedge \text{sign}(\Delta r^s) = \text{sign}(\Delta r^d))$$

$$\text{compatible} = h\_ok \wedge v\_ok \wedge (\text{dst}_i \neq \text{dst}_j)$$

### 1.4 The Core Tradeoff

For each layer, the optimizer faces a genuine tradeoff:
- **Don't move atoms** → zero reconfig cost, but potentially $\chi > 1$ (expensive gate execution)
- **Move atoms to optimal geometry** → $\chi = 1$ is always achievable (e.g., all sources on one row, destinations on another), but reconfig moves cost parallel groups

The do-nothing baseline for Map 2 costs 18 (3 layers × 3 gate groups × 2). The theoretical lower bound is 6 (3 layers × 1 gate group × 2, with zero reconfig). The best known solution is 12. The optimal may not achieve $\chi = 1$ on every layer — the reconfig cost to get there could exceed the gate execution savings.

### 1.5 Key Structural Insight

For the problem instances under consideration (≤4 gates per layer, 5×5 board), **there always exists a placement achieving $\chi = 1$**. Proof by construction: place one atom of each gate on row 0 and the partner on row 1; all displacement vectors are identical → trivially AOD-compatible. This means $\chi > 1$ is never structurally forced; it's always a consequence of the current placement. However, achieving $\chi = 1$ may not be worth the reconfig price, so **both the gate cost and reconfig cost surrogates matter**.

---

## 2. The Surrogate Design Problem

### 2.1 What We Need

A model outputs logits $\pi_\theta(a \mid q, t)$ — for each layer $t$ and relevant atom $q$, a distribution over board cells (after masking illegal positions). We need a **differentiable loss** $\mathcal{L}(\pi_\theta)$ such that:

1. Minimizing $\mathcal{L}$ pushes $\pi_\theta$ toward assignments that minimize the true cost
2. $\mathcal{L}$ is differentiable w.r.t. $\theta$ (for gradient-based optimization)
3. $\mathbb{E}[\mathcal{L}]$ is an unbiased estimator of the true cost (or at minimum, monotonically correlated)

### 2.2 Priorities (Ranked)

1. **Unbiasedness** — the surrogate's expectation matches the true cost
2. **Low variance** — stable gradients for optimization
3. **Computational speed** — must run in the inner training loop
4. **Tightness** — surrogate ≈ true cost, not just correlated

### 2.3 Assumptions

- Per-atom placement distributions are **independent** (factorized joint) given the layer
- Hard constraints (no two atoms on same cell) handled by **masking before** the surrogate, not by the surrogate itself
- Current scope: **5×5 board, ~12 qubits, 3-4 layers, ≤4 gates per layer**
- Framework: **PyTorch**

### 2.4 The Differentiability Challenge

The true cost has two deeply non-differentiable components:

| Component | Non-differentiable operations | Challenge |
|-----------|------------------------------|-----------|
| AOD compatibility | `sign()`, equality checks | Discontinuous per pair |
| Chromatic number $\chi$ | Integer-valued, NP-hard in general | Discrete-valued function of all edges jointly |
| Canonicalization | $\min$ over $2^M$ direction choices | Discrete argmin |
| Reconfig grouping | Same as $\chi$ but on reconfig moves | Same structure |

---

## 3. Approach Evolution

### 3.1 First Attempt: Edge-Independent Enumeration (v1)

**Idea:** For $M$ gates with $\binom{M}{2}$ edge pairs, compute the marginal probability of conflict for each edge independently. Then enumerate all $2^{\binom{M}{2}}$ conflict graph configurations, weight each by the product of its edge probabilities, and look up $\chi$.

**Pipeline:**
```
p_q(cell)  →  P(compat_ij) per gate pair [O(C^4)]  →  P(edge_config) = ∏ P(e_ij)  →  E[χ] = Σ P(e)·χ(e)
```

**Critical flaw:** Edges sharing a gate are correlated (they depend on the same atoms' positions), but the factorization $P(\mathbf{e}) = \prod P(e_{ij})$ assumes independence. This introduced significant bias (~0.2–0.4 cost units) and, more critically, **gradient misalignment** (cosine similarity 0.11 against finite-difference ground truth).

**Test Results (Map 2, Layer 0):**

| Test | Result | Issue |
|------|--------|-------|
| One-hot recovery | 5.94 vs true 6.0 | Soft-min temperature bias |
| Bias (soft dists) | +0.23 to +0.36 | Edge independence assumption |
| Gradient alignment | cos_sim = 0.11 | **Fundamentally broken** |
| Optimization | 6→2 | Works but finds infeasible solutions (atom collisions) |

**Diagnosis:** The optimization "worked" (Test 4) because finding $\chi = 1$ is a broad, easy-to-find basin — even noisy gradients can roll downhill toward it. But the gradient misalignment means this surrogate cannot be trusted for the subtle reconfig-vs-gate tradeoff where precise gradient signal matters.

**Key lesson:** Forward-pass bias and gradient bias are different things. A moderate forward-pass bias (~5%) caused near-total gradient decorrelation because the edge-independence factorization creates artificial gradient pathways that don't reflect the true correlated sensitivity structure.

### 3.2 Root Cause Analysis: Why Edge Independence Fails

For gates $i, j, k$ on 4 distinct atoms each, edges $(i,j)$ and $(i,k)$ both depend on gate $i$'s atom positions. When atom positions shift:
- The true gradient captures how moving gate $i$'s atoms simultaneously changes **both** edges $(i,j)$ and $(i,k)$
- The independent model treats each edge's sensitivity separately, missing the correlated response

In the true cost, shifting gate $i$'s atoms might simultaneously resolve conflicts with gates $j$ and $k$ — a coordinated benefit. The independent model underestimates this coordination because it computes $\partial P(e_{ij})/\partial p$ and $\partial P(e_{ik})/\partial p$ separately, missing that the same atom movement drives both.

### 3.3 Current Approach: Exact Gate-State Enumeration (v2)

**Key insight:** Gates form a matching (no shared atoms between gates in the same layer). Therefore **gate states are independent**. A "gate state" $s_g = (c_{a_g}, c_{b_g})$ is the pair of cells occupied by gate $g$'s two atoms, with domain size $D = C^2 = 625$ for a 5×5 board.

The conflict between gates $i$ and $j$ is a **deterministic function** of their states: $e_{ij} = f(s_i, s_j)$, captured in a precomputable $D \times D$ boolean table. Crucially, the chromatic number $\chi$ is a function of all 6 edges, which are in turn a function of all 4 gate states. By summing over the joint gate-state space directly, we avoid the edge-independence assumption entirely.

**Exact computation:**
$$\mathbb{E}[\chi] = \sum_{s_0, s_1, s_2, s_3} \prod_g P(s_g) \cdot \chi(\{f_{ij}(s_i, s_j)\}_{i<j})$$

where $P(s_g) = p_{a_g}(c_{a_g}) \cdot p_{b_g}(c_{b_g})$ and the sum is over all $D^4 = 625^4 \approx 1.5 \times 10^{11}$ joint states.

**Why this works:**
- **Zero bias** — no approximation, we sum over the exact joint distribution
- **Fully differentiable** — each $P(s_g)$ is a product of placement probabilities; $\chi$ values are precomputed constants; the whole thing is a weighted sum of constants, which is a polynomial in the placement probabilities
- **Captures all correlations** — edge co-occurrences are naturally handled because we enumerate gate states, not edges

**Implementation via variable elimination:**

Direct $D^4$ enumeration is borderline. We use variable elimination with the structure:

```
For each s0 (loop, 625 iters — skip if p0[s0] ≈ 0):
  For each s1 (loop, 625 iters — skip if p1[s1] ≈ 0):
    Vectorize over (s2, s3):
      - Compute 6-bit edge config for all (s2, s3) pairs   → (625, 625) int tensor
      - Look up χ for each config                           → (625, 625) float tensor  
      - Weight by p2[s2] * p3[s3] and sum                   → scalar
    Accumulate with p0[s0] * p1[s1]
```

This is $O(D^2)$ iterations of $O(D^2)$ vectorized work = $O(D^4)$ total, but with:
- Early termination (skip near-zero probabilities, especially effective for sharp distributions)
- Fully vectorized inner loop (the $625 \times 625$ operations are batched tensor ops)
- Precomputed compatibility tables (6 tables of size $625 \times 625$, computed once)

**Expected runtime:** For near-one-hot distributions, most outer loop iterations skip → fast. For soft distributions, worst case ~minutes on CPU, ~seconds on GPU. Acceptable for validation; may need optimization for inner-loop training.

**Precomputation costs:**
- 6 compatibility tables × 4 direction combos = 24 tables of $625 \times 625$ booleans ≈ 9.4M entries, ~9.4 MB
- χ lookup table: 64 entries for M=4

### 3.4 Parallel Track: REINFORCE Baseline (v2-MC)

As a parallel investigation, we're testing REINFORCE with batch-mean and moving-average baselines. This provides:
- **Unbiased gradients** by construction (no structural approximations)
- **Gradient quality benchmark** to compare against the exact analytical surrogate
- **Practical baseline** — if REINFORCE with enough samples gives good gradient signal, it's a simpler path

Tests in progress:
- **Test A:** REINFORCE gradient self-consistency (do independent gradient estimates agree with each other?) across different sample counts K
- **Test B:** REINFORCE gradient vs finite-difference ground truth (does REINFORCE point in the right direction?)
- **Test C:** REINFORCE optimization (can it find good placements?)

### 3.5 Canonicalization Strategy

For the exact surrogate, canonicalization (choosing move direction per gate to minimize $\chi$) is handled by evaluating all $2^M = 16$ direction assignments and taking the minimum. For differentiability:
- **Hard argmin + straight-through:** Pick the best direction, backprop through it. Biased (ignores directions close in cost) but simple.
- **Soft-min:** $-\tau \log \sum \exp(-\text{cost}_d / \tau)$. Smooth but introduces temperature-dependent bias.
- **Exact marginalization:** For each joint gate-state assignment, compute the optimal direction and weight by probability. Zero bias but requires enumerating $D^4 \times 2^M$ terms.

The exact surrogate naturally supports the third option: inside the $D^4$ loop, for each $(s_0, s_1, s_2, s_3)$, compute $\chi$ under all 16 directions and take the min. This adds a factor of 16 but remains exact.

---

## 4. Test Results Summary

### 4.1 Edge-Independent Enumeration (v1) — Completed

| Test | Status | Key Finding |
|------|--------|-------------|
| One-hot recovery | ~PASS (off by 0.055) | Soft-min τ bias, fixable |
| Bias measurement | **FAIL** (+0.23 to +0.36) | Edge independence too strong |
| Gradient alignment | **FAIL** (cos_sim = 0.11) | Gradients essentially uncorrelated with truth |
| Optimization | PASS (6→2) but infeasible | Finds χ=1 basin but ignores collisions |

**Verdict:** Abandoned as primary surrogate. Forward-pass bias is tolerable but gradient misalignment is fatal for the precision needed in reconfig-vs-gate tradeoffs.

### 4.2 Exact Gate-State Enumeration (v2) — VALIDATED

**Full enumeration (CPU, top_k=25 = no pruning):**
- Zero bias confirmed at all sharpness levels (within 2σ of MC)
- Gradient cosine similarity: **1.000000** (machine precision match with numerical FD)
- Speed: ~200-500s per eval on CPU — too slow for training

### 4.3 GPU-Accelerated Sparse Enumeration (v3) — VALIDATED

Fully vectorized 4D tensor computation on GPU with top-K pruning per atom.

**Speed (CUDA, no canonicalization):**

| top_k | Tensor size | Time | 
|-------|------------|------|
| 5 | 390K | 6ms |
| 8 | 16.7M | 7ms |
| 10 | 100M | 24ms |
| 12 | 430M | 88ms |

**With canonicalization (16× direction passes):**

| top_k | σ=4.0 | σ=10.0 |
|-------|-------|--------|
| 8 | 67ms | 61ms |
| 10 | 287ms | 264ms |
| 12 | 1.15s | 1.07s |

**Bias from top-K pruning (σ=4.0, GT=6.72):**

| top_k | Value | Bias | min_kept_mass |
|-------|-------|------|---------------|
| 5 | 6.20 | -0.52 | 0.55 |
| 8 | 6.31 | -0.40 | 0.61 |
| 10 | 6.46 | -0.25 | 0.66 |
| 12 | 6.50 | -0.21 | 0.70 |

At σ=10 (near-deterministic), bias is <0.005 at all top_k. Bias vanishes as distributions sharpen during training.

**Gradient:** Cosine similarity 1.000000 (inherits from v2, pruning is differentiable).

**Optimization:** 6→2 in 56s (0.28s/step), no collisions. Found physically reasonable placements.

### 4.4 REINFORCE Baseline — COMPLETED

**Key finding: the cost landscape is nearly flat along individual logit directions.** Perturbing a single atom's logit by ±0.5 changes the expected cost by ~0.03, while MC standard deviation is ~0.37. This means:

- FD "ground truth" is unreliable: two independent FD estimates have cosine similarity of only 0.08-0.36
- REINFORCE gradient alignment test was comparing signal against noise — the "failures" were inconclusive
- REINFORCE self-consistency requires K≥1024 samples (cos_sim=0.70)
- REINFORCE optimization still works (6→2) but slowly, with collisions

**Verdict:** The exact analytical surrogate is strictly superior — it resolves the tiny gradient signals that MC methods cannot.

---

## 5. Architecture: How the Pieces Fit Together

```
Model outputs logits π_θ(a | q, t) for each atom q in layer t
         │
         ▼
    Masked softmax → p_q(cell) placement distributions
         │
         ├──────────────────────────┐
         ▼                          ▼
    Gate Cost Surrogate        Reconfig Cost Surrogate
    (exact gate-state enum)    (not yet implemented)
         │                          │
         ▼                          ▼
    2 · E[χ_t]                 E[G_t^reconfig]
         │                          │
         └──────────┬───────────────┘
                    ▼
              Layer cost = E[G_t^reconfig] + 2·E[χ_t]
                    │
                    ▼
              Sum over layers → Total loss L(θ)
                    │
                    ▼
              ∇_θ L → update model parameters
```

---

## 6. Next Phases

### Phase 1: Validate Exact Surrogate (Current)

Run the exact gate-state enumeration tests:
1. One-hot sanity → must match true cost exactly
2. Soft bias measurement → must show zero bias
3. Gradient alignment → must show high cosine similarity
4. If validated: benchmark runtime for training-loop feasibility

### Phase 2: Validate REINFORCE Baseline (Current, Parallel)

Measure gradient quality and optimization ability of pure REINFORCE. This gives us a fallback and a comparison point.

### Phase 3: Reconfig Cost Surrogate ← NEXT

**Architecture clarification (Interpretation A):** The model outputs all layers' placement distributions simultaneously — $p_q^{(t)}(\text{cell})$ for every atom $q$ and layer $t$. These are absolute position targets, not sequential decisions. After gate execution in layer $t$, atoms return to their pre-gate positions (round-trip), so the "source" for layer $t
s reconfig is the position chosen for layer $t-1$ (or the initial position for $t=0$).

The reconfig cost for layer $t$ is the number of parallel groups needed to move all relevant atoms from their layer $(t-1)$ positions to their layer $t$ positions. Both source and destination are soft distributions.

**Key differences from gate cost:**
- Reconfig "moves" are per-atom (not per-gate-pair), so $N$ moves instead of $M$ gates
- Each move has soft source AND soft destination: atom $q
s move is drawn from $p_q^{(t-1)} \to p_q^{(t)}$
- No canonicalization (direction is fixed: old position → new position)
- No-ops (atoms that don't move) contribute zero reconfig cost

**Approach:** Same gate-state enumeration principle. An atom's "reconfig state" is $(c_{\text{src}}, c_{\text{dst}})$ with probability $p_q^{(t-1)}(c_{\text{src}}) \cdot p_q^{(t)}(c_{\text{dst}})$. For the first layer, $p_q^{(0-1)}$ is one-hot at the initial position, reducing to $O(C)$ per atom.

**Challenge:** With $N$ movers instead of $M=4$ gates, the conflict graph has up to $\binom{N}{2}$ edges. For $N=8$ relevant atoms: $\binom{8}{2} = 28$ edges, $2^{28} \approx 268M$ conflict graph configurations. The 4D tensor trick doesn't directly apply. Will need either grouping heuristics, MC for this component, or the edge-independence approximation (which may be less biased for reconfig since atoms don't share structure the way gates do).

### Phase 4: Combined Multi-Layer Loss

With Interpretation A (all layers at once), the total loss is:
$\mathcal{L} = \sum_t \left[ \mathcal{L}_t^{\text{reconfig}}(p^{(t)}, p^{(t-1)}) + 2 \cdot \mathcal{L}_t^{\text{gate}}(p^{(t)}) \right]$
where $p^{(t)}$ and $p^{(t-1)}$ are independent model outputs, making the full loss differentiable end-to-end with no sequential sampling.

### Phase 5: Collision Handling

Add soft exclusion constraints ensuring the joint placement is approximately valid (at most one atom per cell). Options:
- Sinkhorn-style doubly-stochastic projection
- Penalty term: $\lambda \sum_c (\sum_q p_q(c) - 1)_+^2$

### Phase 6: Scaling

For larger instances ($M > 6$ gates, bigger boards):
- Gate-state domain $D = C^2$ grows quadratically with board size
- $D^4$ enumeration becomes intractable for $C > 25$
- Transition to: MC with exact-surrogate control variate, penalty-based differentiable coloring, or learned cost predictors
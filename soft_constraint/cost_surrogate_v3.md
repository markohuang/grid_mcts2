# Feasibility-Based Gate Cost Surrogate

## 1. The Physical Constraint

### 1.1 Setup

We have $M$ two-qubit gates in a layer, each pairing atoms $(a_g, b_g)$ on an $H \times W$ grid. To execute a gate, one atom physically moves to the other's position via AOD laser, then returns. The AOD sweeps entire rows/columns simultaneously, creating a fundamental constraint: **two gate moves can only execute in parallel if their displacement vectors don't cause laser-path crossings.**

### 1.2 The AOD Compatibility Condition (Exact)

Two moves $i = (s_i \to t_i)$ and $j = (s_j \to t_j)$, where $s, t \in \mathbb{Z}^2$ are grid positions, are parallel-compatible iff:

**Column axis:** The relative column ordering is preserved from source to destination.
$$h\_ok = \begin{cases}
\text{true} & \text{if } \Delta c^s = 0 \wedge \Delta c^d = 0 \\
\text{false} & \text{if } \Delta c^s = 0 \oplus \Delta c^d = 0 \\
\text{sign}(\Delta c^s) = \text{sign}(\Delta c^d) & \text{otherwise}
\end{cases}$$

where $\Delta c^s = c^s_i - c^s_j$ (source column difference), $\Delta c^d = c^d_i - c^d_j$ (destination column difference).

**Row axis:** Identical condition on rows.

**No collision:** $t_i \neq t_j$ (distinct destinations).

**Full condition:** $\text{compat}(i,j) = h\_ok \wedge v\_ok \wedge \text{nocol}$

### 1.3 Chromatic Number

The minimum number of sequential AOD operations needed is $\chi(G)$, the chromatic number of the conflict graph $G$ where nodes are moves and edges connect incompatible pairs. The gate cost is $2\chi$ (round-trip: enter + exit).

### 1.4 Canonicalization

Gates are symmetric: gate $g = (a_g, b_g)$ can be executed as $a_g \to b_g$ or $b_g \to a_g$. The optimal cost minimizes over all $2^M$ direction assignments: $\chi^* = \min_{\mathbf{d} \in \{0,1\}^M} \chi(G(\mathbf{d}))$.

---

## 2. The Feasibility Reformulation

### 2.1 Key Observation

For our problem instances, $\chi_{\text{gate}} = 1$ is always achievable — there always exists a placement and direction assignment making all gates execute in a single parallel group. Good solutions always achieve this. Therefore:

> **The gate cost optimization reduces to a feasibility problem:** find placements and directions such that all gate moves are pairwise AOD-compatible.

The cost is either 2 (feasible, one round-trip) or $2k$ for $k \geq 2$ (infeasible, multiple round-trips). We need a differentiable surrogate that is minimized when feasibility is achieved.

### 2.2 Decomposition: Direction Constraints are 2-SAT

For gate $g$ with atoms at column positions $\alpha_g = \text{col}(a_g)$, $\beta_g = \text{col}(b_g)$, under direction $d_g \in \{0,1\}$:

$$\text{source}_g^c = \alpha_g(1 - d_g) + \beta_g d_g, \quad \text{dest}_g^c = \beta_g(1 - d_g) + \alpha_g d_g$$

For two gates $i, j$, the four direction combinations produce source/destination differences:

| $(d_i, d_j)$ | source diff | dest diff |
|---|---|---|
| $(0,0)$ | $\alpha_i - \alpha_j$ | $\beta_i - \beta_j$ |
| $(0,1)$ | $\alpha_i - \beta_j$ | $\beta_i - \alpha_j$ |
| $(1,0)$ | $\beta_i - \alpha_j$ | $\alpha_i - \beta_j$ |
| $(1,1)$ | $\beta_i - \beta_j$ | $\alpha_i - \alpha_j$ |

**Critical structural observation:** $(0,0)$ and $(1,1)$ produce the same pair of differences (swapped), and $(0,1)$ and $(1,0)$ produce the same pair (swapped). Since $\text{sign}(x) = \text{sign}(y) \iff \text{sign}(y) = \text{sign}(x)$, the compatibility result is identical for swapped pairs.

Therefore, for each gate pair on each axis, **only the parity $d_i \oplus d_j$ matters**, not the individual directions. The constraint is either:
- "Same direction works" ($d_i = d_j$)
- "Opposite direction works" ($d_i \neq d_j$)
- "Both work" (unconstrained)
- "Neither works" (structurally infeasible for this pair)

This is exactly a **2-SAT** problem: each pairwise constraint is a clause over two binary variables. 2-SAT is solvable in $O(M^2)$ time and has well-studied continuous relaxations.

---

## 3. The Surrogate: Probability of Feasibility

### 3.1 Definition

For each direction assignment $\mathbf{d} \in \{0,1\}^M$ and each gate pair $(i,j)$, define:

$$P_{ij}(\mathbf{d}) = P\left(\text{col\_compat}(i,j \mid d_i, d_j) \;\wedge\; \text{row\_compat}(i,j \mid d_i, d_j) \;\wedge\; \text{nocol}(i,j \mid d_i, d_j)\right)$$

where the probability is over the random atom positions drawn from the soft placement distributions.

The probability that ALL gates can execute in one group under direction $\mathbf{d}$:

$$F(\mathbf{d}) = P\left(\bigwedge_{i < j} \text{compat}(i,j \mid \mathbf{d})\right)$$

The probability of feasibility (over all directions):

$$F^* = \max_{\mathbf{d}} F(\mathbf{d})$$

The surrogate gate cost:

$$\boxed{\mathcal{L}_{\text{gate}} = 2 + \lambda \cdot (1 - F^*)}$$

When $F^* = 1$: all gates guaranteed compatible → cost is exactly 2 (optimal). When $F^* < 1$: some probability of incompatibility → penalty drives optimization toward feasibility.

### 3.2 Computing $P_{ij}(\mathbf{d})$

Each pairwise compatibility probability decomposes by axis:

$$P_{ij}(\mathbf{d}) = P(\text{col\_compat}_{ij} \mid d_i, d_j) \cdot P(\text{row\_compat}_{ij} \mid d_i, d_j) \cdot P(\text{nocol}_{ij} \mid d_i, d_j)$$

**Note:** This factorization across axes assumes column compatibility and row compatibility are **independent conditional on the atom positions**. In fact, they share the same positions, so this is the **marginal independence approximation**. We analyze the bias in Section 4.

**Column compatibility** for a specific direction $(d_i, d_j)$ involves 4 atoms' column positions. Using coordinate marginals (distribution over column values $v \in \{0, \ldots, W-1\}$):

$$P(\text{col\_compat}) = \sum_{v_1, v_2, v_3, v_4} P(v_1) P(v_2) P(v_3) P(v_4) \cdot \mathbb{1}[\text{sign}(v_1 - v_3) = \text{sign}(v_2 - v_4)]$$

where $(v_1, v_2)$ are the source and destination columns for gate $i$ under direction $d_i$, and $(v_3, v_4)$ for gate $j$. This is $O(W^4)$ per pair per axis per direction.

**Row compatibility:** identical, $O(H^4)$.

**No collision:** $P(t_i = t_j) = \sum_c P(t_i^c = c) \cdot P(t_j^c = c)$ for each cell $c$. This requires the full 2D destination distribution, not just marginals. $O(C)$ per pair.

### 3.3 Computing $F(\mathbf{d})$ — The Edge Independence Approximation

The exact joint feasibility is:

$$F(\mathbf{d}) = P\left(\bigwedge_{i < j} \text{compat}_{ij}\right)$$

This is NOT the product of marginals because compatibility events for pairs sharing a gate are correlated (they depend on the same atoms).

We approximate:

$$\hat{F}(\mathbf{d}) = \prod_{i < j} P_{ij}(\mathbf{d})$$

This is the **edge-independence approximation** applied to the specific case of the all-compatible event. See Section 4 for bias analysis.

### 3.4 Optimization over Directions

For $M = 4$: enumerate all $2^4 = 16$ assignments, take the max.

$$\hat{F}^* = \max_{\mathbf{d} \in \{0,1\}^M} \hat{F}(\mathbf{d})$$

For differentiability, use soft-max:

$$\hat{F}^*_\tau = \tau \log \sum_{\mathbf{d}} \exp(\hat{F}(\mathbf{d}) / \tau)$$

Or, since only the argmax matters for the gradient (straight-through): compute all $\hat{F}(\mathbf{d})$, pick the best, differentiate through it.

For larger $M$: exploit the 2-SAT structure. The optimal direction assignment can be found in $O(M^2)$ via implication graph analysis, avoiding the $2^M$ enumeration.

### 3.5 Complexity

$$O\left(2^M \cdot \binom{M}{2} \cdot (W^4 + H^4 + C)\right)$$

For $M = 4, W = H = 5, C = 25$: $16 \times 6 \times (625 + 625 + 25) = 122\text{K}$ ops.

Compare to exact $\chi$ enumeration: $O(2^M \cdot K^{2M}) = 16 \times 64^4 = 268\text{M}$ ops.

**Speedup: ~2000×.**

---

## 4. Bias Analysis

### 4.1 Source of Bias: Edge Independence

The approximation $\hat{F}(\mathbf{d}) = \prod P_{ij}$ treats compatibility events as independent. In reality, events sharing a gate are positively correlated (the same atom positions that make one pair compatible tend to make adjacent pairs compatible too).

**Direction of bias:**

$$\hat{F}(\mathbf{d}) = \prod_{i<j} P_{ij} \leq F(\mathbf{d}) = P\left(\bigwedge_{i<j} \text{compat}_{ij}\right)$$

The product **underestimates** the true feasibility probability. This means the penalty $(1 - \hat{F})$ **overestimates** the infeasibility. The surrogate is **conservative**: it penalizes infeasibility more than necessary.

**Proof sketch:** By the FKG inequality, if the compatibility events are positively correlated (which they are, since favorable atom positions help multiple pairs), then $P(\bigwedge A_i) \geq \prod P(A_i)$.

**Consequence for optimization:** The optimizer works harder than necessary to achieve feasibility, but never falsely declares a configuration feasible when it isn't. This is safe — it might slightly over-penalize configurations near the boundary, but it won't miss the optimum.

### 4.2 Source of Bias: Axis Independence

We compute $P(\text{compat}_{ij}) = P(\text{col\_compat}) \cdot P(\text{row\_compat}) \cdot P(\text{nocol})$, treating axes as independent. In reality, the same atom positions determine both axes.

**Direction of bias:** This is less clear-cut. The row and column positions of an atom on a grid are **functionally related** ($r = c \text{ div } W$, $c = c \text{ mod } W$), so the axis events are not independent. For atoms placed uniformly, the correlation is weak. For concentrated distributions, the correlation is strong but the bias vanishes (both probabilities approach 0 or 1).

**Mitigation:** At the optimum ($P = 1$ on each axis), the bias is zero regardless of correlation. The bias matters only during optimization, where it acts as an approximate penalty.

### 4.3 Exactness at the Optimum

**Claim:** $\hat{F}^* = 1 \iff F^* = 1$ (the approximation is exact at the boundary).

**Proof:** ($\Rightarrow$) If $\hat{F}^* = \prod P_{ij} = 1$ for some $\mathbf{d}$, then $P_{ij} = 1$ for all pairs. Since each $P_{ij}$ is exact (computed from marginals, no approximation in the pairwise case), all pairs are compatible with probability 1, so $F^* = 1$.

($\Leftarrow$) If $F^* = 1$, then all pairs are compatible with probability 1 under some direction. This means $P_{ij} = 1$ for all pairs (since a conjunction of events has probability 1 iff each event has probability 1). So $\hat{F}^* = \prod 1 = 1$.

**This is the same exactness-at-optimum property as the reconfig surrogate ($I = 0 \iff \chi = 1$).** The surrogate is exact precisely where it matters most.

### 4.4 Gradient Behavior

The gradient of $\hat{F}(\mathbf{d}) = \prod_{i<j} P_{ij}$ with respect to an atom's placement logits is:

$$\frac{\partial \hat{F}}{\partial \theta} = \sum_{i<j} \frac{\partial P_{ij}}{\partial \theta} \prod_{k \neq (i,j)} P_k$$

This has a natural **bottleneck-focusing** property: if one pair has $P_{ij} \ll 1$ while others have $P_{kl} \approx 1$, the gradient is dominated by $\partial P_{ij} / \partial \theta$ (the other terms in the product are $\approx 1$). The optimizer automatically focuses on the hardest-to-satisfy constraint.

At the optimum ($P_{ij} = 1$ for all pairs), the gradient is zero — which is correct, since no further improvement is possible. The gradient is nonzero whenever any pair is not fully compatible, providing signal to fix it.

**Potential issue:** The product of many probabilities can be numerically tiny, causing vanishing gradients. For $M = 4$ with 6 pairs, if each has $P_{ij} = 0.9$, then $\hat{F} = 0.9^6 = 0.53$ — still fine. For $M = 8$ with 28 pairs: $0.9^{28} = 0.05$ — getting small. **Mitigation:** Work in log space: $\log \hat{F} = \sum \log P_{ij}$.

---

## 5. Connections to Established Theory

### 5.1 FKG Inequality and Positive Association (Probability/Statistics)

The bias analysis relies on the **FKG inequality** (Fortuin, Kasteleyn, Ginibre 1971): for increasing events on a lattice, $P(A \cap B) \geq P(A) P(B)$.

Our compatibility events are "increasing" in the sense that atom configurations making one pair compatible tend to make others compatible too (the atoms are in a "good arrangement"). This is formalized through the lattice structure of the configuration space.

The FKG inequality guarantees that our edge-independence approximation is a **lower bound** on the true feasibility probability. This is a well-known tool in statistical mechanics (correlation inequalities) and reliability theory (system reliability bounds).

**Related work:** Bounds on system reliability via inclusion-exclusion and FKG — see Barlow & Proschan, "Statistical Theory of Reliability and Life Testing" (1975). Our feasibility probability is analogous to a **series system reliability** where all components (pairwise compatibilities) must function.

### 5.2 2-SAT and Constraint Satisfaction (Computer Science)

The direction assignment problem is a **2-SAT** instance. The literature on 2-SAT relaxations includes:

- **SDP relaxation of MAX-2-SAT:** Goemans-Williamson style rounding gives 0.878 approximation. Our continuous direction variables $x_g \in [-1, +1]$ are analogous.
- **Belief propagation on 2-SAT:** Message-passing algorithms that compute marginal probabilities of satisfying assignments. Our $\hat{F}(\mathbf{d})$ computation is similar to a single BP iteration.
- **Probabilistic 2-SAT:** Random instances have phase transitions. Our problem is structured (not random) but the tools transfer.

**Key reference:** Aspvall, Plass, Tarjan (1979) — "A Linear-Time Algorithm for Testing the Truth of Certain Quantified Boolean Formulas." The implication graph structure of 2-SAT enables $O(M^2)$ direction optimization.

### 5.3 Order-Preserving Maps and Lattice Theory (Mathematics)

The AOD compatibility condition is equivalent to **coordinate-wise order preservation** — the source-to-destination map must be monotone in each coordinate independently.

This connects to:

- **Lattice homomorphisms:** A map $f: L_1 \to L_2$ between lattices that preserves the partial order. The grid $\mathbb{Z}^2$ with the product order is a lattice. AOD compatibility asks whether the source→destination map is a lattice homomorphism restricted to the relevant atoms.

- **Birkhoff's representation theorem:** Finite distributive lattices are isomorphic to lattices of lower sets of finite posets. The set of compatible move configurations forms a distributive lattice.

- **Dilworth's theorem:** The minimum number of chains needed to partition a poset equals the maximum antichain size. Applied to the ordering of moves on each axis, this gives $\chi_{\text{axis}} = \text{LDS length}$ (longest decreasing subsequence).

- **Young tableaux and RSK correspondence:** The Robinson-Schensted-Knuth correspondence maps a permutation to a pair of standard Young tableaux. The length of the first row equals the longest increasing subsequence; the number of rows equals the longest decreasing subsequence (= $\chi_{\text{axis}}$). For random placements, the distribution of $\chi$ is governed by the Tracy-Widom distribution.

### 5.4 Scheduling Theory (Operations Research)

The parallel move scheduling problem is a variant of **interval scheduling** or **job scheduling with conflicts:**

- **Interval graph coloring:** If the conflict graph were an interval graph, $\chi$ would equal the clique number $\omega$ (interval graphs are perfect). AOD conflict graphs are not interval graphs in general, but the axis decomposition shows they are "close" — each axis produces a comparability graph.

- **Parallel machine scheduling:** Scheduling $M$ jobs on parallel machines with pairwise incompatibilities. The minimum number of machines = chromatic number. This is well-studied in OR; see Leung (2004), "Handbook of Scheduling."

- **Graph coloring relaxations:** The fractional chromatic number $\chi_f(G)$ and Lovász theta function $\vartheta(G)$ provide continuous relaxations of $\chi$. Our feasibility approach sidesteps these by targeting $\chi = 1$ directly, which is equivalent to checking whether the conflict graph has an independent set of size $M$ (i.e., the graph is empty under some direction assignment).

### 5.5 Probabilistic Combinatorial Optimization (ML/Optimization)

Our approach of computing $P(\text{feasible})$ from soft distributions and using it as a differentiable loss connects to:

- **Erdős Goes Neural (Karalias & Loukas 2020):** Differentiable penalty functions for combinatorial constraints including graph coloring. Our per-pair $1 - P_{\text{compat}}$ terms are analogous to their edge-based penalties.

- **CombOptNet (Paulus et al. 2021):** Differentiable optimization layers that solve combinatorial problems in the forward pass. Our feasibility check is a lightweight version of this for the specific structure of AOD constraints.

- **Gumbel-Sinkhorn (Mena et al. 2018):** Differentiable relaxations for permutation problems. Our direction assignment is a simpler binary (not permutation) problem, but the principle of differentiable discrete optimization is shared.

- **Straight-through estimators (Bengio et al. 2013):** Using hard decisions in the forward pass with gradient passed through the soft version. Our max-over-directions + differentiate-through-argmax is this pattern.

### 5.6 Monotone Maps and Transportation (Mathematics)

The condition that the source-to-destination map be coordinate-wise monotone is related to **optimal transport** with monotone constraints:

- **Brenier's theorem:** The optimal transport map (Wasserstein-2) between two distributions on $\mathbb{R}^n$ is the gradient of a convex function — which is monotone. However, Brenier monotonicity is with respect to the full vector ordering, not coordinate-wise.

- **Knothe-Rosenblatt rearrangement:** A coordinate-wise monotone transport map. This is exactly the structure AOD requires — each coordinate is transported monotonically, independently. The Knothe-Rosenblatt map is not optimal in the Wasserstein sense but satisfies the AOD constraint by construction.

**Potential insight:** If we constrained the atom placement to be a Knothe-Rosenblatt rearrangement of the initial configuration, feasibility would be guaranteed by construction. The optimization would then be over the space of monotone coordinate maps, which is a convex set.

### 5.7 Potential Pitfalls

**From statistics — multiple testing:** Computing $\hat{F} = \prod P_{ij}$ for many pairs is similar to requiring multiple hypotheses to simultaneously hold. The product can be overly conservative (Bonferroni-like). For $M = 4$ (6 pairs), this is mild. For $M = 10$ (45 pairs), the product could be vanishingly small even when most pairs are compatible.

**From optimization — flat gradients near optimum:** When $\hat{F} \approx 1$, the gradient $\partial(1 - \hat{F})/\partial\theta$ is small. The optimizer may slow down as it approaches feasibility. Mitigation: use $-\log \hat{F}$ instead of $1 - \hat{F}$, which has gradient $-\sum (\partial P_{ij}/\partial\theta) / P_{ij}$ — this amplifies the signal from low-probability pairs.

**From combinatorics — direction correlations:** We maximize $\hat{F}$ over directions independently per pair. But the optimal global direction assignment might not correspond to the per-pair optimal. The 2-SAT structure means global satisfiability can be checked in linear time, but we're currently enumerating (fine for $M \leq 5$, needs 2-SAT solver for larger).

---

## 6. Unified Framework: Gate + Reconfig as Joint Feasibility

### 6.1 The Joint Optimization

The per-layer cost $\text{cost}(t) = \chi_{\text{reconfig}}^{(t)} + 2 \cdot \chi_{\text{gate}}^{(t)}$ involves two monotonicity conditions on the same placement $\mathbf{p}^{(t)}$:

- **Reconfig:** The map $\mathbf{p}^{(t-1)} \to \mathbf{p}^{(t)}$ must be coordinate-wise monotone (on moving atoms). Direction is fixed.
- **Gate:** There must exist a direction assignment $\mathbf{d}$ such that the gate moves under $\mathbf{d}$ are coordinate-wise monotone. Direction is free (canonicalization).

Both are instances of the same mathematical object: checking whether a set of displacement vectors on a grid satisfies coordinate-wise order preservation.

### 6.2 Compositional Consistency

The composition of coordinate-wise monotone maps is coordinate-wise monotone. This means: if every layer's reconfig is individually feasible ($\chi = 1$), the reconfig constraints across layers are automatically consistent. No multi-layer interaction penalty is needed for reconfig — each layer's constraint is independent.

However, the gate constraints at different layers impose different requirements on atom positions (different gates are active), and these compete with the reconfig constraints through the shared positions.

### 6.3 The Unified Surrogate

$\mathcal{L} = \sum_{t} \left[ \mathcal{L}_{\text{reconfig}}^{(t)} + \mathcal{L}_{\text{gate}}^{(t)} \right]$

where:

$\mathcal{L}_{\text{reconfig}}^{(t)} = P(\text{any atom moves}) + \lambda_r \cdot (-\log \hat{F}_{\text{reconfig}}^{(t)})$

$\mathcal{L}_{\text{gate}}^{(t)} = 2 + \lambda_g \cdot (-\log \hat{F}_{\text{gate}}^{(t)})$

$\hat{F}_{\text{reconfig}}^{(t)} = \prod_{i<j, \text{ both move}} P(\text{reconfig compat}_{ij}^{(t)})$

$\hat{F}_{\text{gate}}^{(t)} = \max_{\mathbf{d} \in \{0,1\}^M} \prod_{i<j} P(\text{gate compat}_{ij}^{(t)} \mid \mathbf{d})$

Each pairwise $P(\text{compat})$ is computed identically — from coordinate marginals via the $O(V^4)$ monotonicity check. The only difference is:
- Gate: has direction freedom (enumerate $2^M$ or solve 2-SAT)
- Reconfig: no direction freedom, but weighted by $P(\text{both move})$

### 6.4 The Knothe-Rosenblatt Connection

The Knothe-Rosenblatt rearrangement is the unique coordinate-wise monotone transport map. It guarantees AOD feasibility by construction.

**Theoretical value:** Proves the feasibility set is non-empty and convex (monotone maps form a convex cone). This means the feasibility landscape has no spurious local minima — any configuration can be continuously deformed toward feasibility.

**Practical limitation:** Parameterizing monotone maps differentiably is awkward on a discrete grid. The cumulative-softmax approach (parameterize non-negative increments, cumulative-sum to get monotone sequence) works in 1D but is complex in 2D. The penalty-based approach (allow any placement, penalize infeasibility) is more practical.

**Useful as initialization:** Start the optimization from a Knothe-Rosenblatt rearrangement of the initial positions. This guarantees reconfig feasibility at initialization, giving the optimizer a feasible starting point.

---

## 7. Summary

| Property | Gate Cost | Reconfig Cost |
|----------|-----------|---------------|
| **What it checks** | All gate moves parallel-compatible under some direction | All reconfig moves parallel-compatible |
| **Direction freedom** | Yes ($2^M$ or 2-SAT) | No (fixed src→dst) |
| **Pairwise computation** | $O(V^4)$ per pair per axis | $O(V^4)$ per pair per axis |
| **Direction search** | $O(2^M)$ enumeration or $O(M^2)$ 2-SAT | N/A |
| **Edge-independence bias** | Conservative (FKG: underestimates feasibility) | Conservative (same) |
| **Exactness at optimum** | $\hat{F} = 1 \iff \chi = 1$ | $\hat{F} = 1 \iff \chi = 1$ |
| **Total complexity** | $O(2^M \cdot M^2 \cdot V^4)$ | $O(N^2 \cdot V^4)$ |
| **Log-form gradient** | Bottleneck-focusing: $-\sum \nabla\log P_{ij}$ | Same |
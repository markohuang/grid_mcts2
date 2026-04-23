# Simulated Annealing Baseline

A metaheuristic baseline for per-layer atom placement. Fills the quality/runtime gap between greedy heuristics (`kouhei_policy`) and optimal SMT (`smt_policy`).

## Literature

- **Kirkpatrick, Gelatt, Vecchi (1983)** — *Optimization by Simulated Annealing*, Science 220(4598). Canonical SA. State space + neighbor function + temperature schedule; accept worse neighbors with probability `exp(-ΔC / T)`, cool `T` over time to escape local minima.
- **Chen & Chang (2005)** — *Fast Simulated Annealing for VLSI floorplanning*, TCAD 25(4). Introduces a **three-phase schedule**: (1) high-T random exploration, (2) low-T pseudo-greedy local search, (3) temperature re-heat / hill-climb to escape local minima before final cool-down. Much faster convergence than classic geometric cooling.
- **Tan, Bochen, et al. (2024)** — *Enola: Compilation for Dynamically Field-Programmable Qubit Arrays with Efficient and Provably Near-Optimal Scheduling*, [arxiv 2405.15095](https://arxiv.org/abs/2405.15095). Applies Fast-SA to the **placement** sub-problem of NAQC compilation: given a fixed gate schedule, assign qubits to atom sites minimizing a weighted sum of Euclidean distances between interacting qubits. Cost function is a **distance proxy**, not an AOD parallel-group count.

## What SA does

Given a cost function over a discrete state space, SA performs a biased random walk:

```
s ← initial_state
T ← T_init
for step in range(max_steps):
    s' ← perturb(s)           # sample a neighbor
    ΔC ← cost(s') − cost(s)
    if ΔC < 0 or rand() < exp(−ΔC / T):
        s ← s'                # accept
    T ← cool(T, step)         # monotone non-increasing
return best_seen
```

High `T` → near-uniform random walk (exploration). Low `T` → near-greedy (exploitation). The cooling schedule trades breadth for depth. Fast-SA's re-heat phase adds a controlled escape from local minima found during the low-T phase.

## Adaptation to this problem

**Per-layer decomposition.** Same scope as `smt_policy.solve_layer`: for each gate layer, fix prev positions of all qubits, search for new positions of `relevant_atoms` (qubits appearing in the layer's gate pairs). Greedy across layers, metaheuristic within each layer.

**State.** `{q: (row, col) for q in relevant_atoms}`. Non-relevant qubits stay fixed — they constrain the empty-cell set.

**Neighbors.** Two move types, sampled uniformly:
- *Reassign*: pick a relevant qubit, move it to a random empty cell.
- *Swap*: pick two relevant qubits, swap their cells.

Swap moves are crucial on dense maps where free cells are scarce.

**Cost function — the key departure from Enola.** Enola uses `Σ w_g · dist(q, q')` (distance proxy). This baseline uses the **exact env cost**:

```
layer_cost = count_groups(reconfig_moves) + 2 * count_groups(gate_moves, canonicalize=True)
```

from `neutral_atoms/moves.py` — same function `smt_policy` optimizes over. No proxy. Directly comparable to MCTS / SMT costs.

**Schedule.** Geometric cooling by default (`T *= decay` each step). Optional Fast-SA three-phase schedule gated on a flag for ablation.

**Termination.** Fixed step budget (e.g. 2000 per layer). Returns the best-seen assignment, not the final state.

## Relation to other baselines

| Baseline | Search | Cost | Runtime (5×5 layer) | Optimality |
|---|---|---|---|---|
| `kouhei_policy` | 1-step greedy | exact | <10 ms | myopic |
| `sa_policy` (new) | metaheuristic | exact | ~1 s | strong local optimum |
| `smt_policy` | Z3 `Optimize` | exact | 1 s – 30 min | provably optimal (chromatic number) |
| MCTS + network | learned tree search | dense reward over trajectory | seconds (inference) | depends on training |

`dpqa_policy` is a different axis (global gate re-ordering via SMT, then kohei placement); orthogonal to SA.

## Expected role in evaluation

- **Sanity check** for MCTS: on small maps where SMT gives the optimum, SA should usually match or close the gap. If MCTS can't beat SA on small maps, something is wrong with training.
- **Scaling baseline**: on 8×8 maps (Maps 3/4) where SMT times out, SA gives a meaningful non-optimal target.
- **Ablation anchor**: SA shares the exact cost function with MCTS, so any MCTS gain over SA is attributable to learning / long-horizon planning, not to cost-model differences.

## File

`baselines/sa_policy.py`. Same `plan(initial_positions, tasks, rows, cols) -> atom_viz_plan` signature as existing baselines. Self-contained — imports only `count_groups` and `gates_to_moves` from `neutral_atoms/`.

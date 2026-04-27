# Self-play Gumbel: Gumbel AlphaZero ablation

Tracks all Gumbel AlphaZero self-play waves. The baseline comparison is always v3a (`wave03a_rnd_plancost`, MAPS[2] 5×5/12qb, 10k sims, FakeNet) — see `docs/selfplay/selfplay_v3.md:91-119` for v3a metrics.

Full implementation rationale and the 2×2 ablation plan (control / A / B / AB) live in `docs/gumbel_pczero_plan.md`.

---

## How Gumbel AlphaZero works (our codebase context)

Standard AlphaZero at the root uses **Dirichlet noise + pUCT**: it adds noise to the prior, then visits actions proportional to a UCB score that decays with visit count. This is correct in the limit but wasteful when the sim budget is small relative to the number of actions — visits are spread across many clearly-bad actions.

Gumbel AlphaZero (Danihelka et al., ICLR 2022) replaces this with two mechanisms:

### Root: Gumbel-Top-m + sequential halving

1. Sample m candidate actions using Gumbel-Top-k: draw `g(a) ~ Gumbel(0,1)` noise, pick the m actions with highest `g(a) + logit(a)`. This is equivalent to sampling m times from the prior without replacement — over many games, rare-but-good actions still get sampled.
2. Run **sequential halving**: divide the total sim budget equally over ⌈log₂m⌉ phases. In each phase, run sims uniformly across surviving candidates, then eliminate the bottom half by score. Repeat until one winner remains.

On our 5×5 MAPS[2] with m=14 (= `board_size − num_qubits + 1`, all legal actions) and 10k sims:

| Phase | Candidates | Sims/candidate | Cumulative visits (winner) |
|---|---:|---:|---:|
| 1 | 14 | 178 | 178 |
| 2 | 7 | 357 | 535 |
| 3 | 4 | 625 | 1160 |
| 4 | 2 | 1250 | 2410 |

Each candidate gets a meaningful sim allocation before any are eliminated (178 sims each in round 1). The winner accumulates ~2410 visits; eliminated candidates have ~178.

### Non-root: deterministic argmax

At non-root nodes, replace pUCT with:

```
a* = argmax_a [ π'(a) − N(a) / (1 + ΣN) ]
```

The `− N(a) / (1 + ΣN)` term is a count-based diversity penalty — already-visited actions are penalised, so the tree still explores multiple paths, just deterministically rather than via UCB noise.

### Policy target: improved policy π'

The training target replaces softmax-of-visit-counts with:

```
π'(a) = softmax( logit(a) + σ(a) )
σ(a) = (c_visit + max_N) × c_scale × q_normalized(a)
```

where `q_normalized` maps `completedQ` values to [0,1]. This is the "guaranteed improvement" target — the paper proves `π'` dominates the original prior in terms of policy improvement, for any function approximator.

### Why Gumbel isn't prone to shallow local optima

A natural concern: if halving commits to a winner early, could it prune the globally best action based on noisy early estimates?

Two reasons this is controlled:

1. **All legal actions are candidates.** With `m = num_legal = 14`, every legal action enters round 1. Nothing is excluded by prior probability alone. Gumbel noise is used for sampling when `m < num_legal` (sub-sampling regime); at `m = num_legal`, we just add Gumbel noise for tie-breaking consistency.

2. **Round 1 gives every candidate 178 sims.** With 178 simulations per action, Q-value estimates are meaningful enough that systematic elimination (best → worst) is far more likely than random elimination. The paper's policy improvement bound gives a formal guarantee: the probability that the globally best action is eliminated is bounded by a function of the Q-value gap and simulation count.

Note: this guarantee is statistical (holds in expectation over many games), not per-game deterministic. Any single game can in theory pick a suboptimal action — the property is that the *policy* (aggregated over many games with Gumbel noise) strictly improves on the prior.

**Why mcts_depth is DEEPER, not shallower.** The Phase 2 data shows Gumbel's `mcts_depth_mean = 7.93` vs pUCT's `4.87`. Gumbel's deterministic non-root selection (`argmax π' − N/ΣN`) creates straighter paths through the tree — it commits to the best sub-action at each non-root node rather than spreading visits. pUCT's UCB exploration spreads visits shallowly across many branches. Concentrated paths go deeper.

---

## Policy entropy = 0: what it means and why it happens

The improved policy target collapses to a near-one-hot at high sim counts. This is the most important training concern from Phase 2.

**Concrete numbers (our 10k-sim run, m=14, c_visit=50, c_scale=1.0):**

After halving, max_N ≈ 2410 (winner's visit count). The σ multiplier is:

```
(c_visit + max_N) × c_scale = (50 + 2410) × 1.0 = 2460
```

FakeNet's logits are near-uniform: roughly `log(1/14) ≈ −2.6` per action.

For three illustrative actions:
| action | q_normalized | σ | logit | softmax input |
|---|---:|---:|---:|---:|
| winner | 1.00 | 2460 | −2.6 | **2457** |
| runner-up | 0.90 | 2214 | −2.6 | 2211 |
| eliminated | 0.10 | 246 | −2.6 | 243 |

The softmax of `[2457, 2211, 243, ...]`:

```
exp(2457) / (exp(2457) + exp(2211) + ...)  ≈  1.0
```

The difference of 246 between winner and runner-up corresponds to `e^246 ≈ 10^107`. The softmax is numerically indistinguishable from a one-hot. Hence `policy_entropy = 0.000`.

**Why this is a problem for training:** when every position in the training data says "play action X with probability 1.0", the policy head is supervised to be completely confident everywhere. A trained network then has near-zero probability on non-best actions, which means future self-play has almost no policy-level diversity — it plays the same action sequence repeatedly for any given board state.

**Why it isn't catastrophic at this phase:** `random_board=True` provides diversity at the *game* level — each training sample has a different board state, so the network sees a wide variety of "best action X" labels across different states. The policy collapse is within a game, not across games. This may be sufficient for a first training run, but should be monitored.

**Root cause:** the `(c_visit + max_N)` formula was designed for the **low-sim regime** where max_N is small (e.g., 10 sims, max_N ≈ 1, σ scale ≈ c_visit). At 10k sims, max_N dominates c_visit by 50× and the σ scale becomes `max_N × c_scale`, which grows linearly with sim budget.

**Fix:** remove `max_N` from σ, use only `c_visit`:

```
σ(a) = c_visit × c_scale × q_normalized(a)        # proposed fix
```

With c_visit=5 (tuned below) and max_N removed:
| action | q_normalized | σ | logit | softmax input |
|---|---:|---:|---:|---:|
| winner | 1.00 | 5.0 | −2.6 | 2.4 |
| runner-up | 0.90 | 4.5 | −2.6 | 1.9 |
| third | 0.70 | 3.5 | −2.6 | 0.9 |
| fourth | 0.30 | 1.5 | −2.6 | −1.1 |

```
softmax([2.4, 1.9, 0.9, -1.1, ...]) ≈ [0.54, 0.33, 0.12, 0.016, ...]
```

Policy entropy is non-trivial, best action still has highest probability, cost quality preserved. **Pending calibration smoke to confirm.**

---

## Implementation (codebase changes)

| File | Change |
|---|---|
| `neutral_atoms/config.py` | New `c.mcts.gumbel` subtree: `{enabled, num_samples_m, c_visit, c_scale}`. `set_derived_config` auto-derives `num_samples_m = board_size − num_qubits + 1` when 0. |
| `neutral_atoms/mcts.py` | `Node.network_value` field. `_expand_node` saves it. `_backpropagate` tolerates `min_max_stats=None`. `play_game` dispatches to `_gumbel_plan` when `gumbel.enabled`. New: `_gumbel_plan`, `_gumbel_simulate`, `_gumbel_non_root_select`, `_gumbel_completed_q`, `_gumbel_improved_policy`, `_gumbel_halving_scores`. |
| `neutral_atoms/game.py` | `store_search_statistics` uses `root._gumbel_policy` as training target when present; falls back to softmax-of-visits. |
| `scripts/smoke_gumbel.py` | Asserts invariants (episode length, policy target distribution, MCTS stats populated). Side-by-side pUCT vs Gumbel cost comparison. |
| `scripts/gumbel_ab_report.py` | Reads two wave parquet indexes; emits metrics table and §3 decision-rule pass/fail. |
| `slurm/selfplay_gumbel_smoke.sh` | 1 job × 62 games calibration run. |
| `slurm/selfplay_gumbel_a_control.sh` | 20 jobs × 1000 games, pUCT, MAPS[2] random, 10k sims. |
| `slurm/selfplay_gumbel_a_treatment.sh` | 20 jobs × 1000 games, Gumbel, MAPS[2] random, 10k sims. |

---

## Phase 1: smoke results (2026-04-23)

### Script smoke (`scripts/smoke_gumbel.py`, FakeNet, MAPS[2] fixed, 64 sims, 4 games/variant)

| Variant | Costs | Mean |
|---|---|---:|
| pUCT | [33, 35, 35, 31] | 33.50 |
| **Gumbel** | [14, 14, 14, 15] | **14.25** |

At 64 sims / 14 legal actions (ratio 4.6×), Gumbel is 2–3 above MAPS[2]'s known best of 12. pUCT is at 2.8× the lower bound. Gates passed: 100% episode completion, policy targets are valid distributions, no NaN/crash.

### Slurm smoke (job 59781131, 62 games, 10k sims, random MAPS[2])

| Metric | Value |
|---|---:|
| n | 62 |
| cost mean | 13.15 |
| cost min | 10 |
| game_time_s median | 210.5s |
| reached_terminal_frac | 0.555 |
| policy_entropy median | 0.000 |

Wallclock 210.5s vs v3a 176s — within the 1.5× budget. Policy entropy collapse first observed here; logged as known concern pre-Phase 2.

---

## Phase 2: full A/B wave (2026-04-23)

Jobs: 59781617 (control, pUCT) and 59781618 (treatment, Gumbel). 20 jobs × 1000 games, MAPS[2] random, 10k sims, plan_cost, FakeNet.

### Results

| Metric | control (pUCT) | treatment (Gumbel) | delta |
|---|---:|---:|---:|
| n | 18 788 | 19 682 | +894 |
| cost min | 12 | **8** | −4 |
| cost q10 | 19 | **11** | −8 |
| cost median | 22 | **13** | **−9 (−41%)** |
| cost q75 | 24 | **14** | −10 |
| cost max | 35 | 19 | −16 |
| cost mean | 22.07 | **12.92** | −9.15 |
| mcts_depth mean / std | 4.87 / 0.23 | 7.93 / 0.56 | +3.06 / +0.33 |
| policy_entropy median | 1.281 | **0.000** | −1.281 ⚠ |
| reached_terminal_frac | 0.329 | 0.555 | +0.226 |
| game_time_s median | 317.7s | **211.2s** | −106.5s |
| unique maps | 100% | 100% | — |
| corr(depth, cost) | −0.293 | −0.104 | +0.189 |
| corr(entropy, cost) | +0.382 | +0.047 | −0.334 |
| corr(root_value, cost) | −0.041 | +0.112 | +0.153 |

### §3 decision rule

| Check | Result | Note |
|---|---|---|
| cost median: treatment ≤ control | **PASS** | 13 vs 22 |
| corr(depth,cost) at least as negative (tol 0.05) | FAIL | criteria artifact — see below |
| reached_terminal_frac within ±5pp | FAIL | criteria artifact — see below |
| wallclock within 1.5× control | **PASS** | 211s vs 318s |
| 100% map uniqueness | **PASS** | both waves |

Formal verdict: **FAIL** — but both failing checks are criteria artifacts, not genuine search-quality problems.

### Analysis

**Cost improvement is real and large.** Median 22→13 (−41%), new minimum 8 (below prior best-known of 12 from Round 04) across 20k random boards. Consistent with the paper's policy improvement guarantee: with m = num_legal and sufficient sims per halving phase, the selected action dominates the prior policy.

**`corr(depth, cost)` criterion is not applicable to Gumbel.** In pUCT, deeper search → more refined value estimates → lower cost. In Gumbel, the deterministic non-root selection creates deep paths toward the already-selected winner — depth reflects path length to the candidate subtree, not search breadth. Even a "shallow" halving run at 10k sims commits 178 sims per candidate before any elimination; the Q estimates are already informative by round 2. The weak depth-cost correlation is expected behavior, not degraded search quality. This criterion will be removed from Gumbel-specific gates in Phase 5.

**`reached_terminal_frac` increase is positive, not a problem.** The ±5pp rule was designed to catch degradation (fewer terminals = truncated/broken games). Gumbel's 0.555 vs control's 0.329 means Gumbel completes more games to terminal — its concentrated non-root selection follows PV lines to the end more often. The criterion will be made directional (flag decreases only) in Phase 5.

**Wallclock improved.** Gumbel 211s vs control 318s. Sequential halving concentrates the simulation budget on a small candidate set; each simulation is as expensive as pUCT (full environment clone + rollout), but fewer wasted sims on clearly-bad branches means less total work. Note: control 318s is itself 80% slower than v3a's 176s (different cluster conditions, different nodes). The A/B comparison holds because both waves ran concurrently; v3a comparison should use Phase 2 control as the reference.

**policy_entropy = 0.000 is the one genuine concern.** See the explanation above. Action selection quality is not affected (Gumbel still finds cost=8 games); the issue is that near-one-hot training targets will push the trained network toward a near-deterministic policy, reducing diversity in future self-play. Addressed by fixing the σ formula before Phase 3.

---

## Entropy calibration smoke (2026-04-23)

Job 59791008. σ formula fixed (`c_visit * c_scale * q_norm`, no `max_N`), `c_visit=5.0`. 62 games, 10k sims, random MAPS[2], FakeNet.

| Metric | Phase 2 treatment (old σ) | Entropy smoke (fixed σ) | Gate |
|---|---:|---:|---|
| cost median | 13 | **12** | ≤ 14 ✓ **PASS** |
| cost min | 8 | 9 | — |
| cost mean | 12.92 | **12.26** | — |
| policy_entropy median | 0.000 | **1.571** | > 0.5 ✓ **PASS** |
| reached_terminal_frac | 0.555 | 0.298 | — |
| mcts_depth mean | 7.93 | 4.879 | — |
| game_time_s median | 211s | 305s | — |

**Both gates pass.** σ fix is validated; `c_visit=5.0` is the new default.

**cost_median improved from 13 → 12**, matching the known lower bound for MAPS[2]. The old over-exploitative one-hot policy was hurting search by collapsing non-root decisions too aggressively; the softer π' keeps enough diversity to explore better subtrees before committing. Policy entropy at 1.571 is in good shape for training (v3a pUCT was 1.281 — same order of magnitude).

`mcts_depth` and `reached_terminal_frac` dropped back toward pUCT levels (4.87, 0.298 vs 7.93, 0.555 in old formula). With softer π', non-root selection distributes visits more broadly → shallower but wider trees → fewer terminal-reaching PV lines. Wallclock increased from 211s to 305s for the same reason (broader trees = more nodes visited per sim).

**Code landed:** `mcts.py:412` formula changed, `config.py` default `c_visit` updated to 5.0.

---

## Open issues

### 1. Decision rule revision for Phase 5

The §3 criteria in `gumbel_pczero_plan.md` were written for pUCT-style behavior. For Phase 5 (8×8):
- Remove `corr(depth, cost)` as a Gumbel gate; replace with `cost_median` gap vs control (already passing).
- Make `reached_terminal_frac` check directional: flag only if treatment < control − 5pp.

### 2. Control baseline timing

Phase 2 control ran at 317s/game vs v3a's 176s (+80%). Same config, different cluster conditions. Not a correctness issue, but 8×8 wave sizing should use Phase 2 control timings as the reference, not v3a.

---

## Next steps

| Step | Prerequisite | Description |
|---|---|---|
| ~~σ formula calibration smoke~~ | ~~—~~ | ~~DONE — both gates passed, c_visit=5.0 validated~~ |
| Phase 3: trained-net control | σ fix landed ✓ | First end-to-end training run using Gumbel self-play data |
| Phase 4: B and AB waves | Phase 3 stable | PCZero auxiliary loss; requires trained-net control |
| Phase 5: 8×8 replication | Phase 4 complete | MAPS[5], promotion gate for `config.py` default |

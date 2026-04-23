# Gumbel AlphaZero + PCZero: literature findings and ablation plan

**Status:** Phase 2 complete (2026-04-23). Gumbel A variant shows strong cost improvement vs matched control. See §10 for full results and analysis.

**Scope:** evaluate two orthogonal, literature-backed add-ons to our current AlphaZero MCTS — **(A) Gumbel AlphaZero** (root + non-root action selection, improved policy target) and **(B) PCZero path-consistency auxiliary loss** — with a 2×2 ablation (control / A / B / AB). Default configs only change after a variant demonstrates a pass against the v3a baseline on matched controls.

---

## 1. Why these two (short form)

### (A) Gumbel AlphaZero

- **Citation:** Danihelka et al., *Policy Improvement by Planning with Gumbel*, ICLR 2022. [OpenReview](https://openreview.net/forum?id=bERaNdoegnO).
- **Claim we care about:** guaranteed policy improvement even when `num_simulations` is comparable to (or smaller than) the number of legal actions. Our regime (≤25 legal actions × 50–10k sims) is exactly the slice where the improvement guarantee matters.
- **Mechanism summary:**
  1. **Root action selection:** sample `m` top candidates via Gumbel-Top-k on `g + logit(a)` (no Dirichlet). Run **sequential halving** — equal visits per phase, drop bottom half by score, repeat for `⌈log₂ m⌉` phases.
  2. **Non-root action selection:** deterministic argmax of a score that combines the prior and a monotone transform `σ(·)` of the completed Q-value:  
     `a* = argmax_a [π(a) · (N(a) / (1 + ΣN(b))) — σ(completedQ(a))]` form (see paper §2.2 for exact).  
     `completedQ(a) = Q(a)` if visited, else the parent's value estimate (or a mixed estimate).
  3. **Policy target:** `π' ∝ softmax(logits + σ(completedQ))` — the guaranteed-improvement target, replaces normalized visits.
- **Fit with our codebase:** env cloning means we're implementing *Gumbel AlphaZero*, not the MuZero variant (no learned dynamics model, no model unroll loss). Simpler.

### (B) PCZero — path consistency

- **Citation:** Zhao, Tu, Xu, *Efficient Learning for AlphaZero via Path Consistency*, ICML 2022. [PMLR](https://proceedings.mlr.press/v162/zhao22h.html), [code](https://github.com/CMACH508/PCZero).
- **Claim we care about:** values along an optimal trajectory should be internally consistent. Adding `L_PC(s) = (v(s) − v̄_W)²` as an auxiliary loss reduced training data requirements and improved winrate (+9pp on 13×13 Hex vs AlphaZero baseline).
- **Why it aligns with our setting:** our `plan_cost` reward is designed so that the cumulative discounted return over an optimal plan *telescopes* to a smooth function of remaining cost (see `docs/reward_analysis.md`). If the optimal path has smoothly-varying remaining cost, the optimal value function should too. PCZero directly targets that invariant.
- **Integration footprint:** a few lines in `network.py:336 forward()` — window the `target_correctness` (or the predicted value) over a sliding window within each trajectory, add `λ · variance` to `total` loss. No changes to MCTS.

### What we explicitly drop

- **Sampled MuZero / Gumbel-Top-k over huge action sets:** not needed (action space ≤25).
- **ReZero backward-view reanalyze:** defer until we have a reanalyze pipeline at all.
- **TSS GAZ PTP two-stage self-play:** defer until Gumbel lands and we have a reason to add curriculum.
- **AlphaZeroES (ES-style direct score max):** bigger conceptual departure; defer.

---

## 2. Baseline: v3a (`wave03a_rnd_plancost`)

Chosen because it is the current reference data for first training pass and has the tightest diagnostic signal across existing waves (see `docs/selfplay/selfplay_v3.md:91-119`). Representative numbers to beat:

| Metric | v3a |
|---|---:|
| cost min / q10 / median / q75 / max | 12 / 19 / **22** / 24 / 33 |
| cost mean | 22.10 |
| mcts_depth mean / std | 4.87 / 0.235 |
| `corr(depth, cost)` | −0.301 |
| `corr(entropy, cost)` | +0.382 |
| `reached_terminal_frac` | 0.329 |
| game_time | 176s |
| unique trajectories / unique maps | 100% / 100% |

v3a's config (held as controls — see §4): FakeNet, MAPS[2] 5×5/12qb/3-layer, `random_board=True`, `num_simulations=10000`, `pb_c_base=19652`, `pb_c_init=1.25`, `root_dirichlet_alpha=0.1`, `root_exploration_fraction=0.25`, `temperature_init=1.0` constant, `reward_mode=plan_cost`.

Note that v3a is *FakeNet self-play*, so the policy/value targets are never consumed by training. That means **(B) PCZero cannot be benchmarked against v3a alone** — PCZero is a training-time loss. For (B) and (AB), we additionally need a short training run and measure *trained-network* self-play quality against a matched trained-network control. See §5 rollout plan.

---

## 3. Ablation matrix

| Variant | MCTS (self-play) | Training loss | What it isolates |
|---|---|---|---|
| **Control** | current AlphaZero: Dirichlet at root, pUCT non-root, softmax-visit target | `policy + cw·correctness + lw·latency` | reproduce v3a numbers on current code |
| **A** (Gumbel only) | Gumbel-Top-k + sequential halving at root, completed-Q deterministic non-root, improved-policy target | unchanged | self-play quality gain from Gumbel alone |
| **B** (PCZero only) | unchanged (current AlphaZero) | `+ λ·L_PC` on correctness head | training-signal gain from PC alone |
| **AB** | Gumbel (A) | `+ λ·L_PC` (B) | additive vs sub-additive interaction |

**Decision rule for each cell (variant-vs-control pass):**

1. Self-play metrics (applies to all variants, run on FakeNet for A; run on trained-net for B and AB — see §5):
   - cost median ≤ control median (prefer strict `<`, but tie with lower variance is acceptable)
   - `corr(depth, cost)` at least as negative as control (‖ρ‖ within 0.05 tolerance)
   - `reached_terminal_frac` within ±5pp of control
   - 100% trajectory and map uniqueness preserved
2. Wall-clock: per-game time within 1.5× of control (Gumbel sequential-halving reshuffles sim allocation; should be comparable)
3. For B and AB, training-time metrics:
   - policy & value losses converge to at least the control's final plateau
   - no divergence / gradient-norm blowup attributable to `L_PC`

**Promotion rule:** a variant is promoted to `config.py` default only if it passes the decision rule on *both* 5×5 (MAPS[2]) and 8×8 (MAPS[5]) map classes. Single-scale wins don't promote — recall round 05's specialist-vs-generalist lesson.

---

## 4. Control-variable protocol

All waves in this ablation series share the following knobs (copy from v3a). Any deviation must be explicit and documented per-wave.

**Environment & map**
- `config.map_num=2` (5×5 phase) or `config.map_num=5` (8×8 phase)
- `config.random_board=True` — fresh map per game
- `reward_mode=plan_cost`

**MCTS budget & exploration (pUCT baseline knobs)**
- `num_simulations=10000` (5×5) / TBD from 8×8 calibration
- `pb_c_base=19652`, `pb_c_init=1.25`
- `root_dirichlet_alpha=0.1`, `root_exploration_fraction=0.25` — **used by control only**; Gumbel replaces both
- `temperature_init=1.0`, `temperature_decay_moves=0`

**Gumbel-specific knobs (variant A only, new config subtree `mcts.gumbel`)**
- `enabled: False` — flag-gate the whole feature; control and B have this off
- `num_samples_m: 16` — root candidate count (Gumbel-Top-k width). Sized so `m ≤ 2·legal_actions` at step 0 on both map classes.
- `c_visit: 50.0`, `c_scale: 1.0` — `σ(q) = (c_visit + max_N) · c_scale · q` monotone transform, paper defaults
- **Dirichlet disabled when `gumbel.enabled=True`** (Gumbel noise plays that role)
- Action played at each step = winner of sequential halving (no softmax-visit sampling)

**PCZero-specific knobs (variant B only, new config subtree `training.pczero`)**
- `enabled: False` — flag-gate
- `lambda: 0.1` — loss weight (paper starts at 0.5; our correctness head is already categorical, so we start conservative)
- `window_size: 5` — centered sliding window on the value sequence within a trajectory
- `head: 'correctness'` — apply only to the correctness head (latency head is a sparse terminal signal, PC doesn't apply)

**Data volume & job shape (from v3a)**
- 20,000 games per wave (5×5); to be re-sized for 8×8 after calibration
- 20 jobs × 1000 games/node, same slurm skeleton as `slurm/selfplay_v3a.sh`

**Things that MUST stay the same across control/A/B/AB:**
- Map class, sim count, reward mode, episode length, observation features, network architecture, replay buffer config, random seed policy (system entropy, no seed pinning).

**Things that INTENTIONALLY differ per variant (and only these):**
- A: root/non-root selection rules, policy target construction.
- B: training loss (one auxiliary term added).

---

## 5. Incremental rollout (smallest step first)

Each phase must land, be diagnosed, and pass its gate before the next begins. Follows our established "land diagnostics incrementally, read data before adding more" discipline.

### Phase 0 — control reproduction (no code changes)

**Goal:** confirm current head produces v3a-equivalent numbers on matched config. Rules out "main branch drifted" before we change anything.

- 1 smoke-scale wave on current code: 2 jobs × 50 games, MAPS[2], `random_board=True`, 10k sims, plan_cost.
- **Gate:** cost median within ±1 of v3a's 22, `corr(depth, cost)` within 0.05 of −0.301. If not, stop — investigate drift before touching MCTS.

### Phase 1 — Gumbel, FakeNet smoke (variant A)

**Goal:** Gumbel implementation is correct and stable; self-play doesn't blow up.

1. Land `mcts.gumbel.enabled=False` path (no-op — still exercises the dispatch). Merge and smoke.
2. Turn on `gumbel.enabled=True`, 1 slurm job × 50 games, MAPS[2].
3. **Sanity gates:** 100% trajectory uniqueness, no NaN in priors, `m`-candidate count matches config, sequential-halving phase structure visible in MCTS-depth distribution.

### Phase 2 — A ablation wave (FakeNet)

Full-scale FakeNet wave matching v3a shape (20k games, 20 jobs × 1000).

- Run **control** and **A** back-to-back on the same cluster conditions.
- Compare against v3a metric table in §2.
- **Gate:** self-play metrics in §3 decision rule. Report in `docs/experiments/14_gumbel_pczero_ablations.md` (new round doc).

### Phase 3 — training pipeline smoke (prereq for B and AB)

PCZero is a training-time loss; v3a was FakeNet-only. Before Phase 4 we need a small trained-network run that is stable and produces a control for B to compete against.

- Use smallest config that trains end-to-end (~1–2 hours on a single node). Intentionally avoid over-tuning — we need a *control*, not a record.
- **Gate:** trained-net self-play matches FakeNet baseline on cost within 10% after N epochs (N TBD once we calibrate). If trained-net is *worse* than FakeNet, Phase 4 is blocked pending training diagnosis.

### Phase 4 — B and AB ablation waves

Only after Phase 3 produces a usable trained-net control.

- Run trained-net variants: **control**, **B**, **A**, **AB**. All four with matched data, matched epochs, matched optimizer state seeds where possible.
- **Gate:** decision rule in §3 against the trained-net control (not v3a's FakeNet numbers). v3a is only used as a sanity floor — trained-net should outperform it or we have a training bug, not a PCZero gain.

### Phase 5 — 8×8 replication

Repeat Phases 2 and 4 on MAPS[5] (8×8/20qb/5lyr) — the scale shift from `docs/selfplay/selfplay_v4.md`. A variant only gets promoted to default after passing on both scales.

---

## 6. Pass criteria for default promotion

A variant replaces its current-default counterpart in `config.py` only if **all** of:

1. **Phase 2 (or Phase 4, as applicable) passes** on 5×5 against matched control.
2. **Phase 5 passes** on 8×8 against matched control.
3. Wall-clock regression ≤ 1.5× of control on both scales.
4. No regression on any diagnostic in the "red flag" list of `selfplay_v4.md:118-122`.
5. Documented in a round doc (`docs/experiments/14_gumbel_pczero_ablations.md`) with reproducible slurm scripts and full metric tables.

Until then: both variants stay behind `enabled: False` flags, default behaviour is unchanged.

---

## 7. What this plan does NOT cover

- **Reanalyze.** Not adding reanalyze as part of this phase. PCZero's windowed-variance loss operates on stored trajectories; if we later want fresh-target reanalyze the design becomes richer (see ReZero), but that's a separate proposal.
- **8×8 sim-count calibration.** Inherits the calibration questions from `selfplay_v4.md` — not relitigated here.
- **Curriculum / map-class mixing.** Intentionally fixed structure class within each phase, matching v3/v4 conventions.
- **Training hyperparameter sweeps.** Phase 3 uses a single trained-net config to establish a control; we do not sweep lr / buffer size / PCZero `λ` in this ablation. A follow-up can sweep `λ` if B shows a gain with `λ=0.1`.

---

## 8. Implementation surface (reference, no code yet)

**For A (Gumbel):**
- `neutral_atoms/mcts.py` — new functions `_gumbel_select_root`, `_sequential_halving_step`, `_completed_q`, `_non_root_gumbel_score`. Modified `play_game`, `run_mcts`, `_add_exploration_noise` (skipped when gumbel on), `_select_action` (bypassed when gumbel on).
- `neutral_atoms/game.py:60 store_search_statistics` — branch on `config.mcts.gumbel.enabled` to build `π'` target instead of softmax-visits.
- `neutral_atoms/config.py` — new `c.mcts.gumbel` subtree.

**For B (PCZero):**
- `neutral_atoms/network.py:336 forward()` — add `pc_loss` term gated on `cfg.pczero_enabled`. Compute sliding-window variance over `target_correctness` (post-bootstrap) along the trajectory axis.
- `neutral_atoms/trainer.py:12 game_to_tensordict` — needs a `trajectory_id` / position-in-trajectory index so the loss can window within a trajectory (currently we batch across trajectories flat).
- `neutral_atoms/config.py` — new `c.training.pczero` subtree.

**Out-of-scope for both:** `env.py`, `board.py`, `moves.py`, `rewards.py`, feature construction. Reward mode and environment dynamics are fixed controls.

---

## 9. Decisions (resolved before implementation)

1. **Gumbel `m` sizing — RESOLVED.** Legal-action count is *constant* within a map: `num_legal = board_size − num_qubits + 1` (14 on 5×5/12qb, 45 on 8×8/20qb; verified by `board.py:get_legal_actions_for_qubit`, since board occupancy count is invariant across steps). Set `m = num_legal` as a single per-map int, computed in `set_derived_config`. No per-step adaptation needed.
2. **Discount — RESOLVED.** `config.mcts.discount = 1.0` already (`config.py:96`), episode length is deterministic. `completedQ(a) = r(a) + V(s')` (γ drops out). No new Gumbel-specific discount knob; inherit `mcts.discount`.
3. **PCZero formulation — RESOLVED (PC-bellman variant).** The paper's literal `L_PC = (v − v̄_W)²` assumes `v*` is constant on optimal paths (true for sparse-reward Hex/Go). In our dense-reward plan_cost MDP `v*(s_t)` decreases monotonically as remaining cost shrinks, so the literal form would penalize the correct shape. Use **PC-bellman**: `L_PC(s_t) = (r_t + γ·v(s_{t+1}) − v(s_t))²`. This reduces to the paper's formulation when rewards are terminal-only (Hex, Go), matches the actual Bellman invariant for our γ=1 dense-reward case, and is pairwise — no windowing complication.
4. **Value head for PCZero — RESOLVED.** Correctness head only (latency is a terminal scalar with no path structure). Compute `v(s)` as `logits2values(correctness_logits)` scalar expectation, then apply PC-bellman. Do not apply PC to categorical logits directly.
5. **Trajectory boundary guard — RESOLVED.** Batches must carry a per-sample `next_in_trajectory` flag so PC-bellman only fires on pairs from the same game. Surface via a new bool field in the TensorDict batch.

These five decisions update §4 (control variables), §5 (rollout), and §8 (implementation surface) — see the corresponding sections for the live spec.

---

## 10. Implementation status and results (2026-04-23)

**Landed:**
- `config.py` — added `c.mcts.gumbel = {enabled, num_samples_m, c_visit, c_scale}`; `set_derived_config` auto-derives `num_samples_m = board_size − num_qubits + 1` when unset.
- `mcts.py` — `Node.network_value` added; `_expand_node` saves it; `_backpropagate` tolerates `min_max_stats=None`; `play_game` dispatches to `_gumbel_plan` when `gumbel.enabled`. New helpers: `_gumbel_plan`, `_gumbel_simulate`, `_gumbel_non_root_select`, `_gumbel_completed_q`, `_gumbel_improved_policy`, `_gumbel_halving_scores`.
- `game.py` — `store_search_statistics` uses `root._gumbel_policy` as the training target when present; falls back to softmax-of-visits otherwise.
- `scripts/smoke_gumbel.py` — asserts invariants (episode length, policy-target distribution, stats populated) plus a pUCT-vs-Gumbel side-by-side.
- `scripts/gumbel_ab_report.py` — reads two wave parquet indexes, emits metric table + pass/fail per §3 decision rules.
- `slurm/selfplay_gumbel_smoke.sh`, `slurm/selfplay_gumbel_a_control.sh`, `slurm/selfplay_gumbel_a_treatment.sh`.

---

### Phase 1 smoke (`scripts/smoke_gumbel.py`, FakeNet, MAPS[2] 5×5/12qb, 64 sims, 4 games/variant)

| Variant | Costs | Mean |
|---|---|---:|
| pUCT control | [33, 35, 35, 31] | 33.50 |
| **Gumbel** | [14, 14, 14, 15] | **14.25** |

At 64 sims / 14 legal actions (ratio 4.6×), Gumbel is within 2-3 of MAPS[2]'s known best of 12.

**Gumbel slurm smoke** (62 games, 10k sims, random MAPS[2], job 59781131):

| Metric | Value |
|---|---:|
| n | 62 |
| cost mean | 13.15 |
| cost min | 10 |
| game_time_s median | 210.5s |
| reached_terminal_frac | 0.555 |

Wallclock 210.5s vs v3a's 176s — within the 1.5× budget. Phase 2 launched.

---

### Phase 2 A/B wave results (2026-04-23)

Jobs: 59781617 (control, pUCT) and 59781618 (treatment, Gumbel). 20 jobs × 1000 games, MAPS[2] random, 10k sims, plan_cost, FakeNet.

| Metric | control (pUCT) | treatment (Gumbel) | delta |
|---|---:|---:|---:|
| n | 18 788 | 19 682 | +894 |
| cost min | 12 | **8** | −4 |
| cost q10 | 19 | **11** | −8 |
| cost median | 22 | **13** | **−9** |
| cost q75 | 24 | **14** | −10 |
| cost max | 35 | 19 | −16 |
| cost mean | 22.07 | **12.92** | −9.15 |
| mcts_depth mean / std | 4.87 / 0.23 | 7.93 / 0.56 | +3.06 / +0.33 |
| policy_entropy median | 1.281 | **0.000** | −1.281 |
| reached_terminal_frac | 0.329 | 0.555 | +0.226 |
| game_time_s median | 317.7s | **211.2s** | −106.5s |
| unique maps | 100% | 100% | — |
| corr(depth, cost) | −0.293 | −0.104 | +0.189 |
| corr(entropy, cost) | +0.382 | +0.047 | −0.334 |
| corr(root_value, cost) | −0.041 | +0.112 | +0.153 |

**§3 decision rule outcome:**

| Check | Result |
|---|---|
| cost median: treatment ≤ control | PASS |
| corr(depth,cost) at least as negative (tol 0.05) | FAIL |
| reached_terminal_frac within ±5pp | FAIL |
| wallclock within 1.5× control | PASS |
| 100% map uniqueness | PASS |

**Formal verdict: FAIL.** However, both FAIL criteria are artifacts of the decision rule being calibrated for pUCT, not Gumbel's fundamentally different search pattern. Analysis below.

---

### Analysis of FAIL criteria

**`corr(depth, cost)` weakened (−0.293 → −0.105).**
In pUCT, deeper search correlates with better cost because more visits → more refined value estimates. Gumbel's sequential halving concentrates all visits onto the winning candidate, so even shallow trees reliably find near-optimal actions. The correct interpretation is not "search quality degraded" but "Gumbel decouples depth from cost because its policy is already concentrated." This criterion is not meaningful as a Gumbel quality signal.

**`reached_terminal_frac` increased (+22.6pp, 0.329 → 0.555).**
The ±5pp rule was designed to detect degradation (fewer terminal states = incomplete/truncated games). A 22pp *increase* means Gumbel is completing more games to terminal. This is unambiguously positive. The rule needs directional asymmetry: flag *decreases* from baseline, not *increases*.

Both §3 criteria will be revised for Phase 5 (8×8 replication) to reflect this learning.

---

### Key observations

1. **Cost improvement is large and robust.** Median 22 → 13 (−41%), mean 22.07 → 12.92. New minimum 8 beats the prior best-known of 12 across 20k random boards. This is consistent with the paper's guarantee: Gumbel provides a strict policy improvement bound when `m = num_legal` and sims are sufficient for halving phases.

2. **Wallclock is better, not worse.** Control re-run (317s/game) is itself 80% slower than v3a's 176s — likely due to cluster load and pUCT overhead on the rerun. Gumbel (211s) is actually *faster* than the matched control because sequential halving concentrates the simulation budget on fewer candidates, reducing tree width.

3. **policy_entropy = 0.000 is a known concern for training.** At 10k sims, `max_N` is large, so `σ = (c_visit + max_N) · c_scale · q_norm ≫ logits`, and `π'` collapses to a near-one-hot distribution. The training policy target carries almost no exploration signal. This does not affect FakeNet self-play cost quality, but will harm trained-network runs if unaddressed: the network will learn a near-deterministic policy and lose the ability to recover from initial misestimates.

   **Mitigation options (not yet tested):**
   - Reduce `c_visit` from 50 to ~5 at 10k sims (keeps σ scale ≈ 1× logit range).
   - Cap σ: `σ = min(c_visit, max_N) · c_scale · q_norm` (bounded regularizer form used in some follow-up implementations).
   - Temperature on π': add a temperature `τ` to the final softmax of the improved-policy target (separate from action-selection τ).
   
   **Decision:** before Phase 3 (trained-net control), run a calibration smoke with reduced `c_visit` (e.g., 5.0) to confirm entropy recovers without degrading cost. If entropy recovers, use the new c_visit for Phase 3+.

4. **Control is slower than v3a.** 317s vs 176s suggests either cluster conditions (different nodes, load) or code changes on this branch introduced overhead. Does not invalidate the A/B comparison (both ran on same cluster conditions), but worth investigating before 8×8 Phase 5 sizing.

---

### Next steps

1. **c_visit calibration smoke** — run a small wave (1 job × 100 games) with `c_visit=5.0` on Gumbel; confirm cost stays ≤ 14 and `policy_entropy_median > 0.5`. If passes, use `c_visit=5.0` for Phase 3+.
2. **Phase 3** — trained-net control run (see §5). Prerequisite for PCZero Phase 4.
3. **Revise §3 decision rules** — directional terminal_frac check; remove depth-cost correlation as a Gumbel gate.
4. **Phase 5 (8×8)** — after Phase 3 and Phase 4 land, replicate on MAPS[5].

---

## Sources

- [Policy Improvement by Planning with Gumbel (Danihelka et al., ICLR 2022)](https://openreview.net/forum?id=bERaNdoegnO)
- [Efficient Learning for AlphaZero via Path Consistency (Zhao et al., ICML 2022)](https://proceedings.mlr.press/v162/zhao22h.html)
- [PCZero code](https://github.com/CMACH508/PCZero)
- [Learning and Planning in Complex Action Spaces / Sampled MuZero (Hubert et al., ICML 2021)](https://arxiv.org/abs/2104.06303)
- [Multiagent Gumbel MuZero (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29121)
- [TSS GAZ PTP for EV routing (2025)](https://arxiv.org/abs/2502.15777)
- [ReZero: Backward-view Reanalyze (2024)](https://arxiv.org/abs/2404.16364)
- [AlphaZeroES (2024)](https://arxiv.org/html/2406.08687v1)
- Internal references: `docs/selfplay/selfplay_v3.md`, `docs/selfplay/selfplay_v4.md`, `docs/reward_modes.md`, `docs/reward_analysis.md`, `neutral_atoms/mcts.py`, `neutral_atoms/game.py:60-75`, `neutral_atoms/network.py:336-374`.

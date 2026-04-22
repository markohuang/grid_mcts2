# Gumbel AlphaZero + PCZero: literature findings and ablation plan

**Status:** planning. No code changes yet. Ablation design locked to compare against `docs/selfplay/selfplay_v3.md` (v3a, `wave03a_rnd_plancost`) as the single reference baseline.

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

## 9. Open questions to resolve before Phase 1

1. **Gumbel `m` sizing.** Paper uses `m = min(num_simulations, num_actions)` in the limit; our `legal_actions` varies per step. Confirm a single config knob suffices or we need `m` as a function of step.
2. **Gumbel non-root discount.** Paper assumes undiscounted or discount folded into value. Our `discount < 1` and we have per-step dense rewards; verify `completedQ(a) = r(a) + γ·V(s')` matches paper convention.
3. **PCZero window across trajectory boundaries.** Batches mix states from different trajectories; window must not cross game boundaries. Enforce via `trajectory_id` mask.
4. **Value bins interaction with PCZero.** We use categorical value heads (`scalar_to_two_hot`). `L_PC` should operate on `logits2values(correctness_logits)` (scalar mean), not on logits directly. Confirm before implementation.

These go into Phase 1's smoke-test checklist; they're not blockers for writing the plan but they are blockers for writing the first PR.

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

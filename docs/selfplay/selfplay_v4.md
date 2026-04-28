# Self-play v4: 8×8 random-map waves

Continuation of `selfplay_v3.md` (5×5 random maps). v4 scales to **8×8 boards with 20 qubits, 5 layers, 6 gates per layer** — a structurally harder problem class than v3's 5×5/12qb/3-layer setting.

## Scale shift vs v3

| Axis | v3 (MAPS[2]) | v4 (MAPS[5]) | Factor |
|---|---:|---:|---:|
| Board cells | 25 (5×5) | 64 (8×8) | 2.6× |
| Qubits | 12 | 20 | 1.7× |
| Layers | 3 | 5 | 1.7× |
| Gates/layer | 4 | 6 | 1.5× |
| Episode length (confirmed via `_resolve_map`) | 24 steps | 60 steps | 2.5× |
| Action space (board_size) | 25 | 64 | 2.6× |
| Legal actions at step 0 | ~14 | ~45 | 3.2× |
| Uniform-BFS depth at 10k sims | log₁₀(10000) / log₁₀(14) ≈ 3.5 | log₁₀(10000) / log₁₀(45) ≈ 2.4 | shallower |

The search-depth problem compounds: we need MORE sims just to match v3's effective tree depth. And on top of that, episodes are 2.5× longer, so each game takes 2.5× more env-step work per sim budget.

**Status (2026-04-28):** FakeNet data collection complete. Both waves finished. Training pipeline is v4a01 — see [docs/pipelines/v4a01.md](../pipelines/v4a01.md) for live status.

## Open questions answered by the FakeNet waves

- **What cost distribution to expect.** No reference optimum for MAPS[5] (unlike map 2 where Round 04 reported cost=12). First v4 wave defines the baseline.
- **Whether 10k sims (v3a's setting) produces usable data on 8×8.** At branching ~45, uniform-BFS depth is ~2.4. Might be too shallow for meaningful Q-discrimination.
- **Whether 20k sims (2×) is enough, or we need 50k+.**
- **Per-game wallclock.** Rough back-of-envelope: 2.5× episode length × sim-budget-factor × (any non-linear tree growth). Could be 5× to 15× v3a's 176s/game. I don't trust this estimate — need a calibration run.
- **Whether plan_cost still wins at this scale.** v3a confirmed plan_cost > layer_delta on 5×5 random. 8×8 has sparser reward (remaining_cost delta is smaller per step relative to total episode length). Ordering could flip.

## Assumptions for the v4 plan (flag before implementing)

1. **Use new MAPS[5] entry** — `8x8_20qb_6gpl_5lyrs`, verified to produce 60-step episodes with 12 unique qubits per layer. Added to config.py this session.
2. **Inherit v3a's winning config**: `pb_c_base=19652`, `pb_c_init=1.25`, `dirichlet=0.1`, `τ=1.0` constant, FakeNet, `reward_mode=plan_cost` (explicitly passed in slurm, config default not yet adopted).
3. **Start with 20k sims.** 2× v3a's 10k. Compensates partially for higher branching factor. Not clearly optimal — may need 50k+ after calibration.
4. **Smoke test FIRST** before committing to a 20-job production wave. One slurm job, small sample, to measure per-game wallclock and verify the pipeline works on 8×8.
5. **Target ~20k games for v4a production wave** (parity with v3 scale). May need to reduce per-node game count to fit wallclock.
6. **No structure-class variation yet.** v4 holds structure fixed at MAPS[5]; random_board varies the specific qubit placements and gate assignments within that structure. Varying structure (e.g. mixing in MAPS[4] 30qb maps) deferred.

## Known unknowns I'm NOT assuming

- That 8×8 trajectory diversity will be 100% unique. Almost certainly yes (action space is larger), but verify post-wave.
- That `corr(depth, cost)` stays negative on 8×8. Depth being much shallower relative to episode length could weaken this signal.
- That plan_cost still wins. Will test in v4a against layer_delta in later v4b only if v4a calibration completes successfully.
- That reached_terminal_frac is meaningful on 8×8. With 60-step episodes and depth-2.4 trees, it's mathematically unlikely any sims reach terminal. Expect ~0 initially.

## Plan

### Step 1: v4_smoke — calibration run — DONE

Results: pUCT 1392s/game, Gumbel 835s/game. Gumbel chosen for production (−42% cost vs pUCT).
Full results in v4a01.md § "Step 1".

### Step 2: v4a — FakeNet production wave — DONE

v4a_gumbel (20k games, Gumbel, 835s/game) and v4a_puct (19k games, pUCT, 1392s/game) completed 2026-04-27.
Bootstrap data at `/scratch/huang651/grid_mcts2/datasets/v4a_gumbel`. Training pipeline: see v4a01.md.

Measured from v4a_gumbel data:
- **Cost distribution shape.** We want a wide distribution (low rel_spread = narrow = bad, high rel_spread = wide = good). Target: rel_cost_spread > 0.30.
- **Cost min / median.** No reference optimum, so we track relative improvement across future v4 waves.
- **Correlations in the same direction as v3a.** `corr(depth, cost) < 0`, `corr(entropy, cost) > 0`, `corr(noop, cost) < 0`, `corr(root_v, cost) ≤ 0`.
- **`reached_terminal_frac`.** Expected to be LOW on 8×8 (probably < 0.05). Tracks how much compute bypasses the untrained value head. Baseline for future sim-count scaling experiments.
- **Diversity**: 100% unique trajectories expected. 100% unique maps by construction.
- **Policy entropy**: we expect HIGHER than 5×5 (more actions in action space). Max = ln(64) ≈ 4.16. A good number is somewhere around 2.0–3.0 — concentrated but not collapsed. Still a meaningful training target for the NN.

### Step 3: v4b — reward mode A/B (pending v4a results)

After v4a completes, compare to v3a-style pattern. If v4a shows the same search-quality signals as v3a (strong correlations, problem-adaptive depth), proceed to v4b with layer_delta to A/B reward modes at 8×8 scale.

If v4a numbers look broken (correlations near zero, no problem-adaptive search), the issue is likely sim count being too low. In that case, next wave is v4a_high (50k sims same config) before testing layer_delta.

## Metrics to track across the v4 series

Relative metrics (scale-invariant — compare v4 waves to each other):
- `rel_cost_spread = (q90-q10) / median` — target > 0.30
- `corr(depth, cost)` — target negative
- `corr(entropy, cost)` — target positive
- `mcts_depth_std / mcts_depth_mean` — target > 0.03 (problem-adaptive)
- `top-K depth gap / depth_mean` — target > 0.05 (good games searched deeper)

Absolute metrics (map-class-specific — anchored to MAPS[5]):
- `cost_min` — best achievable by search
- `cost_median` — typical search quality
- `reached_terminal_frac` — value-head bypass rate at this sim count
- `policy_entropy` (at τ=1, no decay) — targets for NN to learn

Diversity:
- Actual trajectory uniqueness via hash — should be 100% at n=20k
- Per-map diversity — should be 100% by construction

## What success looks like for v4

**Minimum viable (would accept as warmstart-adjacent data):**
- cost distribution has `rel_spread > 0.25`
- `corr(depth, cost)` meaningfully negative (< −0.1)
- `corr(entropy, cost)` meaningfully positive (> +0.15)
- `corr(noop, cost)` preserved (< −0.2)
- Policy entropy between 1.5 and 3.0 (not collapsed, not uniform)

**Stretch (would be happy with):**
- Same but stronger
- `reached_terminal_frac > 0.05` — any fraction of sims reaching terminal on 8×8
- `mcts_depth_std > 0.15`
- Top-100 depth gap > 0.15

**Red flags (if observed, stop and debug):**
- `corr(depth, cost) > 0` — deeper search predicts WORSE outcomes. Search is broken.
- Policy entropy stuck near uniform (> 3.5 at τ=1) — search not concentrating at all.
- Cost distribution extremely narrow or bimodal — something pathological in reward shape at scale.
- `reached_terminal_frac = 0.00` exactly — no sims reach terminal anywhere in the wave. Indicates branching / sim count mismatch is complete; re-scale before collecting more data.

## What v4 will NOT answer

- Whether trained NN + v4 data converges well (need training pipeline)
- Whether 8×8 generalizes to 15×15 (future scale)
- Whether plan_cost vs layer_delta preference at 8×8 predicts behavior in trained-value regimes
- Optimum cost for MAPS[5] structure class (need a trained solver or oracle)

Done when: v4a lands, diagnostics pass or hit a red flag, v4b decision is made based on data.

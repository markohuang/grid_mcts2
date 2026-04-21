# Round 08: Reward-Signal Error Modes Before Generalist Scale-Up

## Decision

Before committing to a generalist-focused roadmap, determine whether `plan_cost` or `layer_delta` is the better foundation, and identify the actual failure mode rather than optimizing around an unverified story.

## What We Know So Far

### Fixed-map evidence

- `plan_cost` from scratch on map2 reaches cost 12 quickly, but then policy entropy collapses from `2.49 -> ~0.12` while MCTS depth rises from `1.9 -> ~10.4`. Performance remains pinned at 12 for the rest of training.
- `layer_delta` from scratch on map2 also reaches cost 12, does so faster, and keeps entropy healthy around `1.1`, but MCTS depth stays shallow at `~4.5-4.9`.

Interpretation:

- `plan_cost` appears to provide enough cross-layer signal to solve the specialist task, but training converges into a very sharp low-entropy policy.
- `layer_delta` appears to provide a much stronger local learning signal, but may be relying less on deep search and more on a strong local policy.

### Generalist evidence

- `plan_cost` generalist runs on random boards underperform badly: avg cost `~17.8-19.0` at 50 sims, `~15.6-17.1` at 200 sims.
- These runs show low `mcts_reward_frac` (`0.07-0.21`) and modest depth (`~10-11`), consistent with reward-path cancellation across long traversals.
- `plan_cost_unbiased` from scratch does not learn at all on the same task and keeps entropy near-random.

Interpretation:

- We have evidence that `plan_cost` struggles on generalist training.
- We do **not** yet know whether `layer_delta` solves that problem, because the generalist `layer_delta` runs from Round 06 were killed before epoch 1 and no replacement generalist run was completed.

## Main Uncertainties

1. Is `plan_cost` failing on generalist training primarily because cumulative simulation rewards cancel, or because policy/value learning is weak for other reasons?
2. Will `layer_delta` generalize better than `plan_cost`, or will it become too myopic because immediate reward ignores future-layer consequences?
3. Is `layer_delta`'s good specialist behavior caused by better credit assignment, or by a different search regime that may not transfer to novel maps?
4. How much of the current story is real, and how much is an artifact of noisy evaluation (`val_cost_corr` at `n=20` is not reliable enough)?

## Missing Information

These are the main gaps preventing a clean design decision:

1. **Direct generalist comparison is missing.**
   We do not have `layer_delta` vs `plan_cost` on the same random-board setup with identical hyperparameters.

2. **Value calibration measurement is too noisy.**
   `val_cost_corr` at 20 games per epoch is not enough to distinguish "uncalibrated value" from noise.

3. **Reward-cancellation is only inferred indirectly.**
   We track `mcts_reward_frac`, but not the full reward-path statistics:
   - reward sum mean/std
   - sign changes per simulation
   - fraction of sims reaching a layer boundary
   - per-layer reward contribution

4. **Search-vs-policy attribution is missing.**
   We do not know whether a checkpoint fails because:
   - policy prior is poor
   - MCTS cannot discriminate actions
   - value targets are wrong
   - or deeper search is simply not being used effectively

5. **Held-out generalization eval is too narrow.**
   Current eval uses a very small map pool. That is enough for directional evidence, but not enough to make a strong architecture choice.

## Round 08 Principles

- Measure assumptions before changing the algorithm.
- Compare rewards under matched settings.
- Separate specialist error modes from generalist error modes.
- Prefer short diagnostic runs over another large unguided sweep.

## Hypotheses

### H1: `plan_cost` generalist failure is primarily a reward-cancellation problem

Prediction:
- On random boards, `plan_cost` will show lower cumulative-reward discrimination (`mcts_reward_frac`, reward-path variance) than `layer_delta`.
- Increasing sims helps only modestly because the underlying signal is weak/canceling.

Success criterion:
- `layer_delta` produces clearly stronger reward-path diagnostics than `plan_cost` under matched settings.

### H2: `layer_delta` improves generalist optimization but may sacrifice cross-layer reasoning

Prediction:
- `layer_delta` generalist training will learn faster than `plan_cost` early on.
- If it is too myopic, it will improve early metrics but plateau on eval cost or show poor later-layer behavior.

Success criterion:
- `layer_delta` beats `plan_cost` on early and mid-training generalist eval without obvious later-layer regressions.

### H3: current value-calibration conclusions are premature

Prediction:
- With a larger no-noise eval sample, `val_cost_corr` will become interpretable and may differ materially between rewards.

Success criterion:
- 100+ no-noise eval games per checkpoint produce a stable enough calibration estimate to compare runs.

## Required Instrumentation Before or Alongside This Round

These are lightweight and directly tied to the open questions:

1. **Reward-path diagnostics in MCTS**
   Add per-root aggregates:
   - `mcts_reward_sum_mean`
   - `mcts_reward_sum_std`
   - `mcts_reward_abs_sum_mean`
   - `mcts_boundary_reach_frac`

2. **Better eval for calibration**
   Add a checkpoint-eval command/script that can run 100-200 no-noise games on:
   - fixed map2
   - a fixed held-out random-map pool

3. **Optional but high-value**
   Add a policy-only eval mode or low-sim eval mode (`num_simulations=1`) so we can tell whether search is rescuing a weak policy.

Without items 1 and 2, we can still run training experiments, but the interpretation will remain soft.

## Experiment Plan

### Phase A: Minimal instrumentation

Goal:
- improve the quality of diagnosis, not training

Deliverables:
- richer MCTS reward-path metrics
- stable checkpoint evaluation on a fixed held-out map pool

### Phase B: Matched short-horizon training runs

Use one short diagnostic budget first, not 300-epoch runs.

Shared settings:

- map2
- `alpha=0.3`
- `aug=True`
- `seed in {42, 123}`
- `epochs=60`
- `num_selfplay=20`
- `batch_size=128`
- `early_stopping_patience=60`

Launched so far:

| Run | ID | Seed | Status |
|-----|----|------|--------|
| `plan_cost_g` | `e9968d5c` | 42 | running |
| `layer_delta_g` | `153cf066` | 42 | running |

#### B1: Generalist head-to-head

Primary missing comparison.

```bash
# 8A: plan_cost generalist control
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 --config.random_board=True \
  --config.training.epochs=60 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.experiment.curriculum_maps=3 \
  --config.experiment.curriculum_initial_phase=3 \
  --config.experiment.study=round08_reward_error_modes \
  --config.experiment.hypothesis=H1 \
  --config.experiment.variant=plan_cost_g \
  --config.experiment.tags=reward,generalist,control \
  --config.experiment.notes="generalist control for reward comparison"

# 8B: layer_delta generalist treatment
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 --config.random_board=True \
  --config.training.epochs=60 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.experiment.curriculum_maps=3 \
  --config.experiment.curriculum_initial_phase=3 \
  --config.env.reward_mode=layer_delta \
  --config.experiment.study=round08_reward_error_modes \
  --config.experiment.hypothesis=H2 \
  --config.experiment.variant=layer_delta_g \
  --config.experiment.tags=reward,generalist,treatment \
  --config.experiment.notes="direct generalist comparison against plan_cost"
```

Replicate both with `seed=123` before drawing conclusions.

#### B2: Search-sensitivity check

Only run this after B1 if `layer_delta` looks promising or ambiguous.

```bash
# 8C: plan_cost generalist, sims=200
... same as 8A plus --config.mcts.num_simulations=200 \
  --config.experiment.hypothesis=H1 \
  --config.experiment.variant=plan_cost_g_s200

# 8D: layer_delta generalist, sims=200
... same as 8B plus --config.mcts.num_simulations=200 \
  --config.experiment.hypothesis=H2 \
  --config.experiment.variant=layer_delta_g_s200
```

Purpose:
- test whether one reward benefits materially more from extra search
- distinguish "bad local signal" from "insufficient search budget"

### Phase C: Checkpoint diagnostics

At epochs `10, 30, 60` for each run:

- run 100-200 no-noise eval games on fixed map2
- run 100-200 no-noise eval games on a fixed held-out random-map pool
- compute:
  - avg / std / best cost
  - `val_cost_corr`
  - `avg_mcts_depth`
  - `mcts_reward_frac`
  - reward-path diagnostics from Phase A

## Decision Table

### If `layer_delta` beats `plan_cost` on generalist eval and keeps healthy diagnostics

Decision:
- move forward with `layer_delta` as the primary generalist reward

Next step:
- focus on restoring any missing cross-layer reasoning rather than replacing the reward entirely

### If `layer_delta` learns faster but generalizes no better or worse

Decision:
- treat it as a specialist-friendly but possibly myopic signal

Next step:
- test hybrid reward ideas or value-target changes only after confirming which later-layer metrics fail

### If `plan_cost` still wins once measured carefully

Decision:
- keep `plan_cost`, but focus on reducing cancellation / improving search discrimination

Next step:
- tune around signal preservation rather than switching rewards

## Minimum Bar For A Strong Conclusion

Do **not** make the reward decision based on:

- one seed
- 20-game calibration estimates
- one lucky best-cost checkpoint
- or fixed-map results alone

Do make the decision if we have:

- direct generalist head-to-head (`plan_cost` vs `layer_delta`)
- at least 2 seeds
- stable no-noise checkpoint eval
- reward-path diagnostics showing whether cancellation is real

## Practical Recommendation

The first thing to do is **not** another broad sweep.
The first thing to do is:

1. add the missing reward-path and stable eval diagnostics
2. run the direct generalist head-to-head
3. only then decide whether the right move is `layer_delta`, `plan_cost`, or a hybrid follow-up

## Outcome Update (2026-04-02)

The instrumentation and the missing direct head-to-head were completed.

### Completed runs

| Run | ID | Seed | Status |
|-----|----|------|--------|
| `plan_cost_g` | `e9968d5c` | 42 | completed |
| `layer_delta_g` | `153cf066` | 42 | completed |

### Direct generalist comparison

At the current 60-epoch diagnostic budget on random 5x5 boards:

- `plan_cost_g` (`e9968d5c`) finished around `avg_cost=17.9`, `avg_mcts_depth=9.3`, `mcts_reward_frac=0.12`, and in-loop eval remained around `18`.
- `layer_delta_g` (`153cf066`) finished around `avg_cost=19.2`, `avg_mcts_depth=2.4`, `mcts_reward_frac=0.90`, and in-loop eval also remained around `18`.

Interpretation:

- `layer_delta` clearly fixes the reward-path cancellation story, but that stronger local signal did **not** translate into better generalist performance in this matched seed-42 run.
- The leading failure mode is no longer "reward too weak"; it is more likely "missing long-horizon search structure."
- `plan_cost` remains the stronger generalist baseline for now, but the direct reward swap alone is not a satisfying answer because its value target remains surrogate-biased.

### Follow-up fixed-map diagnostic that changed the roadmap

To test whether `plan_cost` is most useful as a **search scaffold** rather than a learning target, a fixed 5x5 control/treatment pair was run:

| Run | ID | Setup | Result |
|-----|----|-------|--------|
| `layer_delta_fixed_control` | `2bdbe5b8` | `layer_delta`, no search bonus | finished around `avg_cost=17.8`, `depth=2.4`, `eval_map0_best_so_far=15` |
| `layer_delta_fixed_search_bonus_w2` | `66559890` | `layer_delta` + MCTS-only `plan_cost` bonus, `weight=2.0` | finished around `avg_cost=12.0`, `depth=4.6`, `eval_map0_best_so_far=11` |

This follow-up does **not** prove the generalist problem is solved, but it materially changes the next decision:

- the search-only surrogate produced a large specialist gain
- depth and boundary reach both increased
- the value target stayed on `layer_delta`

### Decision after Round 08

Do **not** promote pure `layer_delta` as the new generalist reward.

Do **not** keep iterating on pure `plan_cost` as if the surrogate target itself were the final answer either.

Instead, launch a 5x5 generalist three-way follow-up:

1. `plan_cost` generalist control
2. `layer_delta` generalist control
3. `layer_delta` + MCTS-only `plan_cost` search bonus

That follow-up is tracked in `09_search_bonus_generalist_5x5.md`.

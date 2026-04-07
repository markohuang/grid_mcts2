# Reward Hybrid Designs

This note turns the current reward discussion into concrete follow-up variants we can implement after the current Round 08 diagnostic pair.

## Why a Hybrid?

Current evidence suggests:

- `plan_cost` provides useful **future-aware shaping**, but it is a surrogate over current positions and ignores future reconfiguration dynamics.
- `layer_delta` is much closer to the true objective, but immediate rewards are mostly **within-layer** and may be too myopic for generalist training.

So the natural next step is not "pick one forever", but:

1. keep an **unbiased or near-unbiased base objective**
2. inject **future-aware shaping** in a controlled way

## Design Goals

Any hybrid should try to satisfy four constraints:

1. Preserve long-horizon structure needed for generalist training
2. Avoid depending entirely on the `plan_cost` surrogate
3. Keep implementation small enough for fast iteration
4. Make diagnostics interpretable

## Variant A: Potential-Shaped `layer_delta`

### Idea

Use corrected `layer_delta` as the base reward, then add a scaled `plan_cost` delta as a shaping term:

```text
r_total = r_layer_delta + λ_plan * r_plan_delta
```

Where:

- `r_layer_delta` is the current corrected unbiased reward
- `r_plan_delta = remaining_surrogate_cost_before - remaining_surrogate_cost_after`

### Why it is attractive

- Lowest implementation risk
- Keeps the actual objective in the reward
- Restores cross-layer guidance directly into search
- Gives us a single interpolation knob `λ_plan`

### Why it is principled

`r_plan_delta` is a difference of a state potential. With `γ=1`, this is potential-based shaping. That means we can add it without changing the optimal policy in the idealized tabular setting, while still changing learning dynamics a lot.

### What it tests

Can we keep `layer_delta`'s strong local signal while borrowing just enough of `plan_cost`'s future-awareness to help generalization?

### Minimal implementation

- Add reward mode: `layer_delta_shaped`
- Reuse existing:
  - `_compute_remaining_cost()` for shaping
  - corrected `layer_delta` boundary correction for base reward
- Add config:
  - `env.plan_shaping_weight`

### Diagnostics to watch

- held-out avg cost
- `avg_mcts_depth`
- `mcts_reward_frac`
- `mcts_boundary_reach_frac`
- entropy by layer

### Recommended sweep

```text
λ_plan ∈ {0.25, 0.5, 1.0}
```

Start with `0.5`.

## Variant B: Unbiased Reward, Surrogate in Search Only

### Idea

Keep environment rewards as `layer_delta` or `layer_completion`, but inject `plan_cost` only inside MCTS scoring.

Examples:

```text
value_score = Q_actual + λ_surrogate * surrogate_bonus
```

or

```text
prior = policy_prior + λ_surrogate * surrogate_prior
```

### Why it is attractive

- Clean separation between objective and heuristic
- Value target stays unbiased
- Lets us test whether `plan_cost` helps mainly as a **search heuristic** rather than a training target

### Why it is riskier

- More invasive in MCTS
- Harder to normalize
- More ways to accidentally distort search

### What it tests

Is the main utility of `plan_cost` in shaping **search**, not in defining the return?

### Recommended form

Start with a small normalized bonus in UCB, not a large prior rewrite.

## Variant C: Annealed Shaping Schedule

### Idea

Start training with shaped rewards, then decay shaping over time:

```text
r_total(t) = r_layer_delta + λ_plan(t) * r_plan_delta
```

with:

```text
λ_plan(0) = 1.0
λ_plan(T) -> 0
```

### Why it is attractive

- Bootstrap with dense future-aware guidance
- Finish closer to the true objective

### Why it is highest risk

- Adds schedule confounds
- Harder to interpret if it works
- Requires enough training stability that annealing actually matters

### What it tests

Does `plan_cost` mainly help **early bootstrapping**, with `layer_delta` better for later refinement?

## Recommended Order

### 1. Variant A: `layer_delta_shaped`

This is the first thing to build.

Why:

- smallest code change
- cleanest test of the "best of both worlds" idea
- diagnostics remain interpretable

### 2. Variant B: search-only surrogate

Only do this if Variant A still fails, but the current Round 08 evidence keeps suggesting that future-aware search structure matters.

### 3. Variant C: annealed shaping

Only do this after we know static shaping helps.

## Concrete Post-Round-08 Plan

If the current `plan_cost` vs `layer_delta` result holds:

1. implement `layer_delta_shaped`
2. run the same Round 08 protocol against:
   - `plan_cost`
   - `layer_delta`
   - `layer_delta_shaped`
3. use the same diagnostics and held-out eval

Suggested first shaped run:

```text
reward_mode = layer_delta_shaped
plan_shaping_weight = 0.5
```

## What Would Count As Success?

`layer_delta_shaped` should:

- beat plain `layer_delta` on held-out generalist eval
- retain better entropy / collapse behavior than pure `plan_cost`
- show deeper search than plain `layer_delta`
- not regress badly on fixed-map performance

## What Would Count As Failure?

`layer_delta_shaped` would be a bad direction if:

- it behaves almost exactly like `plan_cost`, meaning the shaping term dominates
- it behaves almost exactly like `layer_delta`, meaning shaping is too weak
- it destabilizes training and makes the diagnostics harder to interpret

## My Recommendation

After the current Round 08 seed-42 pair finishes and we have checkpoint eval:

- If `plan_cost` is still clearly ahead, do **not** jump straight to search-only surgery.
- Build **Variant A** first.

That gives us the cleanest test of the hypothesis:

> the missing ingredient in `layer_delta` is not stronger local reward, but controlled future-aware shaping.

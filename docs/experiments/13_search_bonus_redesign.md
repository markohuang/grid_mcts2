# Round 13 — Search bonus redesign + diagnostic

## Context

R08 (fixed 5×5) showed `layer_delta + plan_cost_search_bonus_weight=2.0` beat plain `layer_delta` by 6 points (avg 12.0 vs 17.8) and `plan_cost` was a wash (avg 17.9). R09 reproduced a smaller (3-point) version on generalist. Every R10/R12 run inherited `sb=2.0` as fixed — no ablations since.

Since then we fixed two things that plausibly distort R08/R09's picture:
1. **Action sampling** (`a130767`) — `softmax(N/τ)` → `softmax(log N / τ)`. Stops the effective temperature from sharpening with `num_simulations`.
2. **Observation cache** (`6d40fd0`) — `Game.from_dict` no longer hands stale terminal obs to bootstrap targets.

R12 diagnostic also found `root_dirichlet_alpha=0.3` (up from 0.03) significantly improves exploration at 5×5 (R12 `b398d087` hit best=11 by epoch 10).

So the R08 finding that the bonus is worth 3–6 points is on outdated MCTS. We need to re-ablate, and we want to distinguish three possible explanations for why the bonus works:

- **A: one-shot lookahead.** The bonus only matters because at N=0 the child gets a useful heuristic push; after a few visits it's redundant with V_θ's estimate. If true, the legacy `ucb` term and a decayed `ucb_decay` variant should perform the same.
- **B: persistent gravity.** The bonus is doing the work precisely *because* it never decays — it behaves like a permanent prior offset that keeps visits flowing to plan-cost-descending children even after many sims. If true, `ucb_decay` collapses toward `ctrl_ld`.
- **C: nothing.** Post-fix, the gap is gone. The old MCTS was broken in a way the bonus incidentally patched.

Any of A/B/C is informative. A/B both justify the redesign (move the heuristic into the prior — either as a one-shot warm start at expansion or as a proper `pb_c`-enveloped prior mix). C says drop the bonus entirely and move on.

## Code change

Added `config.mcts.plan_cost_search_bonus_mode ∈ {off, ucb, ucb_decay, prior_mix}` (default `off`). The old free-floating `weight > 0` behavior is now `mode='ucb'`.

- `off` — no heuristic injection.
- `ucb` — legacy: `β · child.search_bonus` added to `_ucb_score`. Non-decaying.
- `ucb_decay` — diagnostic: `β · child.search_bonus / (1 + child.visit_count)`. Same initial push, decays with visits.
- `prior_mix` — principled: at `_expand_node`, for each legal action probe a cloned env one step, take `plan_cost_delta / cost_ub`, mix into the log-prior with weight `β`, then softmax. The heuristic then rides the normal `pb_c · √(ΣN)/(1+N)` envelope and distills into π_θ via visit counts.

`track_plan_delta` is auto-enabled in `main.py` iff `mode != 'off'` (explicit, no more silent coupling on `weight > 0`).

See `neutral_atoms/mcts.py:_expand_node`, `_ucb_score`.

## Design

Base config matches R12 `b398d087` (the validated R12 winner) so results slot directly above that run in the timeline:

```
map_num=2 (fixed map2, 5×5)
num_simulations=250
root_dirichlet_alpha=0.3
root_exploration_fraction=0.25
temperature_decay_steps=1000
temperature_final=0.25
network.p_hsize=64
training.num_selfplay=50
training.batch_size=128
training.seed=42
training.data_augmentation=True
experiment.early_stopping_patience=50
epochs=15    # b398d087 hit best=11 at ep10; 15 gives margin
```

## Variants

| Label           | reward_mode  | bonus_mode  | β   | First-order question |
|-----------------|--------------|-------------|-----|----------------------|
| `ctrl_ld`       | layer_delta  | off         | 0   | Does current MCTS already close the R08 gap without any bonus? (C) |
| `ctrl_pc`       | plan_cost    | off         | 0   | Is the shaped-reward route competitive now? |
| `ucb_w2`        | layer_delta  | ucb         | 2.0 | Reproduce R08's +bonus win on new code. |
| `ucb_decay_w2`  | layer_delta  | ucb_decay   | 2.0 | Persistence (B) vs one-shot (A) diagnostic. |
| `prior_mix_w2`  | layer_delta  | prior_mix   | 2.0 | Does the principled formalization match or beat legacy `ucb`? |

## Decision rules

- `ctrl_ld` ≈ `ucb_w2`: the bonus is obsolete (C). Remove it, delete the mode switch.
- `ucb_w2` still wins ≥3 pts over `ctrl_ld`:
  - `ucb_decay_w2` ≈ `ucb_w2` → one-shot (A). Implement prior-mix at root only.
  - `ucb_decay_w2` ≈ `ctrl_ld` → persistence (B). Keep full prior-mix at every expansion, β anneal.
  - `ucb_decay_w2` between → partial both; default to full prior-mix.
- `prior_mix_w2` matches or beats `ucb_w2` → promote to default, delete `ucb`/`ucb_decay` modes.
- `ctrl_pc` beats all layer_delta variants → rethink; potential-based shaping isn't the wrong answer.

## Launch

```bash
./experiments/run_r13_variant.sh <label> <reward_mode> <bonus_mode> <beta> <gpu_id>
```

Two-GPU split:
- GPU0 chain: `ctrl_ld` → `ctrl_pc` → `ucb_decay_w2`
- GPU1 chain: `ucb_w2` → `prior_mix_w2`

## Run IDs

| Variant | Run ID | Status |
|---------|--------|--------|
| ctrl_ld | — | pending |
| ctrl_pc | — | pending |
| ucb_w2 | — | pending |
| ucb_decay_w2 | — | pending |
| prior_mix_w2 | — | pending |

## Results

_Pending._

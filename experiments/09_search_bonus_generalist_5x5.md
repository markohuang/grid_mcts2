# Round 09: 5x5 Generalist Search-Bonus Follow-up

## Decision

Round 08 showed that swapping `plan_cost` for plain `layer_delta` does **not** solve generalist training by itself, but the fixed-map search-bonus follow-up suggests `plan_cost` may be most useful as a search heuristic rather than as the learning target.

This round tests that directly on random 5x5 boards.

## Findings Entering This Round

### Generalist seed-42 diagnostic from Round 08

| Run | ID | Reward / setup | Final read |
|-----|----|----------------|------------|
| `plan_cost_g` | `e9968d5c` | `plan_cost`, random 5x5 | `avg_cost~17.9`, `depth~9.3`, `mcts_reward_frac~0.12` |
| `layer_delta_g` | `153cf066` | `layer_delta`, random 5x5 | `avg_cost~19.2`, `depth~2.4`, `mcts_reward_frac~0.90` |

Interpretation:

- `layer_delta` gives much stronger local reward-path signal.
- That signal alone is not enough to recover the deeper search behavior that the generalist task seems to need.

### Fixed-map search-bonus diagnostic from Round 08 follow-up

| Run | ID | Setup | Final read |
|-----|----|-------|------------|
| `layer_delta_fixed_control` | `2bdbe5b8` | `layer_delta`, fixed 5x5 | `avg_cost~17.8`, `depth~2.4`, best eval `15` |
| `layer_delta_fixed_search_bonus_w2` | `66559890` | `layer_delta` + MCTS-only `plan_cost` bonus | `avg_cost~12.0`, `depth~4.6`, best eval `11` |

Interpretation:

- injecting `plan_cost` only into MCTS looks promising
- the value target stays on `layer_delta`
- the open question is whether that benefit survives the jump from specialist to generalist training

## Questions

1. Does `layer_delta + search_bonus` improve over plain `layer_delta` on random 5x5 boards?
2. Can it close the gap to or beat `plan_cost` while keeping the value target on the less biased reward?
3. If it helps, does it do so by restoring deeper search and more layer-boundary progress, rather than by simply collapsing the policy?

## Hypotheses

### H1: MCTS-only `plan_cost` bonus restores long-horizon search structure to `layer_delta`

Prediction:

- treatment depth and boundary reach rise clearly above plain `layer_delta`
- treatment entropy falls into a purposeful regime without the severe collapse seen in pure `plan_cost` fixed-map runs

### H2: `layer_delta + search_bonus` beats plain `layer_delta` and closes the generalist gap

Prediction:

- treatment improves held-out avg cost versus `layer_delta`
- treatment reaches or beats the `plan_cost` control by epoch 30 or 60

### H3: If search bonus only helps fixed maps, the benefit will disappear on random boards

Prediction:

- treatment may still deepen search, but held-out cost will remain close to plain `layer_delta`
- this would imply the bonus is exploiting specialist structure rather than repairing generalist reasoning

## Design

Three-way matched comparison on random 5x5 boards.

Shared settings:

- map2
- `random_board=True`
- `alpha=0.3`
- `aug=True`
- `epochs=60`
- `num_selfplay=20`
- `batch_size=128`
- `curriculum_maps=3`
- `curriculum_initial_phase=3`
- `early_stopping_patience=60`
- `seed in {42, 123}`

Only the reward / search setup changes:

1. `plan_cost` generalist control
2. `layer_delta` generalist control
3. `layer_delta` + `plan_cost_search_bonus_weight=2.0`

## Commands

```bash
# 9A: plan_cost generalist control
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 --config.random_board=True \
  --config.training.epochs=60 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.experiment.curriculum_maps=3 \
  --config.experiment.curriculum_initial_phase=3 \
  --config.experiment.early_stopping_patience=60 \
  --config.experiment.study=round09_search_bonus_generalist_5x5 \
  --config.experiment.hypothesis=H2 \
  --config.experiment.variant=plan_cost_g_control \
  --config.experiment.tags=reward,search-bonus,generalist,control,plan-cost \
  --config.experiment.parent_run=e9968d5c \
  --config.experiment.notes="round09 5x5 generalist baseline with plan_cost"

# 9B: layer_delta generalist control
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 --config.random_board=True \
  --config.training.epochs=60 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.experiment.curriculum_maps=3 \
  --config.experiment.curriculum_initial_phase=3 \
  --config.experiment.early_stopping_patience=60 \
  --config.env.reward_mode=layer_delta \
  --config.experiment.study=round09_search_bonus_generalist_5x5 \
  --config.experiment.hypothesis=H1 \
  --config.experiment.variant=layer_delta_g_control \
  --config.experiment.tags=reward,search-bonus,generalist,control,layer-delta \
  --config.experiment.parent_run=153cf066 \
  --config.experiment.notes="round09 5x5 generalist baseline with layer_delta"

# 9C: layer_delta + MCTS-only search bonus
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 --config.random_board=True \
  --config.training.epochs=60 --config.training.num_selfplay=20 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.experiment.curriculum_maps=3 \
  --config.experiment.curriculum_initial_phase=3 \
  --config.experiment.early_stopping_patience=60 \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.experiment.study=round09_search_bonus_generalist_5x5 \
  --config.experiment.hypothesis=H2 \
  --config.experiment.variant=layer_delta_g_search_bonus_w2 \
  --config.experiment.tags=reward,search-bonus,generalist,treatment,layer-delta \
  --config.experiment.parent_run=66559890 \
  --config.experiment.notes="round09 5x5 generalist treatment with MCTS-only plan_cost bonus weight=2.0"
```

Replicate all three with `seed=123` before making a final design decision.

## Run IDs

| Run | ID | Seed | Status |
|-----|----|------|--------|
| `plan_cost_g_control` | `e3b850c4` | 42 | running |
| `layer_delta_g_control` | pending | 42 | queued behind `e3b850c4` in batch launcher |
| `layer_delta_g_search_bonus_w2` | pending | 42 | queued behind `e3b850c4` in batch launcher |
| `plan_cost_g_control_s123` | pending | 123 | queued |
| `layer_delta_g_control_s123` | pending | 123 | queued |
| `layer_delta_g_search_bonus_w2_s123` | pending | 123 | queued |

## Decision Rules

Move forward with the search-bonus design only if all of the following hold:

- treatment clearly beats plain `layer_delta`
- treatment search depth and boundary reach improve materially
- treatment is at least competitive with `plan_cost` on held-out cost
- the result survives a second seed

If treatment deepens search but does **not** improve held-out cost, then the next change should probably be reward shaping or value-target work rather than more MCTS-only tuning.

## Monitoring

```bash
python analyze.py \
  <plan_cost_run_id> \
  <layer_delta_run_id> \
  <search_bonus_run_id>
```

Checkpoint eval remains the same as Round 08:

- run deterministic eval at epochs `10`, `30`, and `60`
- compare fixed-map and held-out random-map pools
- use the new reward-path diagnostics to confirm whether the treatment is actually changing search behavior

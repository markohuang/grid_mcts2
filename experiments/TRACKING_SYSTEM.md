# Core MCTS Experiment Tracking System

This repo now has a lightweight two-layer tracking system:

1. **Machine-readable run tracking** in `outputs/<run_id>/`
2. **Human-readable hypothesis tracking** in `experiments/`

The goal is to answer four questions quickly:

1. What hypothesis was this run testing?
2. What changed relative to the control?
3. What outcome would count as evidence?
4. What decision did we make after seeing the result?

## Layer 1: Run Metadata

Every run now saves:

- `config.json`: frozen training config
- `manifest.json`: why the run exists
- `metrics.jsonl`: per-epoch metrics
- `solutions/`: best plan and trace

Use these config fields when launching a run:

```bash
../grid_mcts2/.venv/bin/python main.py \
  --config.experiment.study=round08_reward_signal \
  --config.experiment.hypothesis=H2 \
  --config.experiment.variant=treatment \
  --config.experiment.tags=reward,novel-maps \
  --config.experiment.notes="layer_delta on random boards with alpha=0.3" \
  --config.experiment.parent_run=aa0b9487 \
  --config.experiment.decision="switch default reward if avg_cost improves by >=1.0"
```

Recommended meanings:

- `study`: one decision batch, usually one markdown round
- `hypothesis`: stable question id within the study
- `variant`: control, treatment, ablation-A, sweep-200sims, etc.
- `tags`: short searchable labels
- `notes`: one-line reminder of the point of the run
- `parent_run`: checkpoint or baseline this run depends on
- `decision`: what you plan to do if the evidence is positive

## Layer 2: Hypothesis Docs

Use one markdown round doc per decision batch, plus one tracker file for the active queue.

- `experiments/README.md`: index of completed rounds
- `experiments/core_mcts_tracker.md`: active backlog and decision ledger
- `experiments/<round>.md`: detailed plan, commands, results, analysis

The round doc should be the place where you compare runs and make claims.
The run manifest should be the place where you recover intent for any individual run.

## Canonical Workflow

1. Add or update the hypothesis in `experiments/core_mcts_tracker.md`.
2. Create a new round doc from `experiments/templates/core_mcts_round_template.md`.
3. Launch each run with `study`, `hypothesis`, `variant`, and short `notes`.
4. Use `python analyze.py --study <study>` during execution.
5. After enough evidence, update the round doc with results and mark the tracker entry as confirmed, refuted, inconclusive, or superseded.
6. Record the follow-up decision explicitly.

## Experiment Design Rules

Keep each hypothesis narrow:

- Good: "Increasing `num_simulations` from 50 to 200 improves novel-map avg_cost by >=1.0 at fixed reward mode."
- Weak: "Try more exploration stuff."

Each study should have:

- One control configuration
- One primary success metric
- One maximum runtime budget
- One explicit stop condition

Avoid changing more than one major axis at once:

- Search: `num_simulations`, Dirichlet noise, temperature
- Reward: `reward_mode`, scaling, shaping
- Distribution: fixed map vs random board vs curriculum
- Optimization: batch size, lr, training steps

## Canonical Metric Bundle

For core MCTS studies, prefer this bundle in analysis/writeups:

- `eval_map0_best`
- `eval_map0_avg`
- `avg_cost`
- `avg_mcts_depth`
- `mcts_reward_frac`
- `avg_policy_entropy`
- `val_cost_corr`
- `epochs_completed`
- wall-clock time

Interpretation:

- `eval_map0_*`: decision metric for the reference map
- `avg_cost`: broad training-time behavior
- `avg_mcts_depth`: whether search is actually reaching useful depth
- `mcts_reward_frac`: whether reward signal survives along traversals
- `avg_policy_entropy`: collapse vs healthy exploration
- `val_cost_corr`: rough value calibration check

## Naming Conventions

Use short stable ids.

- `study`: `round08_reward_signal`
- `hypothesis`: `H1`, `H2`, `H3`
- `variant`: `control`, `sims200`, `layer-delta`, `temp-slow`

Keep long prose in `notes`, not in ids.

## Querying Runs

```bash
python analyze.py
python analyze.py --study round08_reward_signal
python analyze.py --hypothesis H2
python analyze.py --tag reward
python analyze.py <run_id>
```

## Decision Status Vocabulary

Use one of:

- `planned`
- `running`
- `confirmed`
- `refuted`
- `inconclusive`
- `superseded`

This keeps the tracker readable and prevents "half-believed" conclusions from lingering.

# Round XX: <Study Title>

## Decision

What repo-level decision is this round supposed to inform?

## Questions

1. H1: <question>
2. H2: <question>

## Success Criteria

- H1 success metric: `<metric>` with threshold `<value>`
- H2 success metric: `<metric>` with threshold `<value>`
- Runtime budget: `<epochs / hours / seeds>`
- Stop condition: `<early stop / fixed epoch budget / confidence threshold>`

## Controls

- Control run:
- Treatment run(s):
- Constants held fixed:

## Commands

```bash
# Control
../grid_mcts2/.venv/bin/python main.py \
  --config.experiment.study=<study_id> \
  --config.experiment.hypothesis=H1 \
  --config.experiment.variant=control \
  --config.experiment.tags=<comma-separated-tags> \
  --config.experiment.notes="<one-line purpose>"

# Treatment
../grid_mcts2/.venv/bin/python main.py \
  --config.experiment.study=<study_id> \
  --config.experiment.hypothesis=H1 \
  --config.experiment.variant=<treatment_name> \
  --config.experiment.tags=<comma-separated-tags> \
  --config.experiment.notes="<one-line purpose>"
```

## Run Table

| Hypothesis | Variant | Run ID | Parent | Status | Key diff |
|------------|---------|--------|--------|--------|----------|
| H1 | control | `<run_id>` | — | running | baseline |

## Results

| Run | best | avg | eval_best | eval_avg | depth | reward_frac | entropy | value_corr | Notes |
|-----|------|-----|-----------|----------|-------|-------------|---------|------------|-------|
| control |  |  |  |  |  |  |  |  |  |

## Outcomes

- H1: confirmed / refuted / inconclusive
- H2: confirmed / refuted / inconclusive

## Decision

- Adopt:
- Reject:
- Next run:

## Notes

- Unexpected behavior:
- Risks to interpretation:
- Missing instrumentation:

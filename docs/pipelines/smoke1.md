# Pipeline smoke1 — infrastructure smoke test

**Pipeline id:** `smoke1`
**Date:** 2026-04-21
**Purpose:** Validate the 3-job chain (pretrain → selfplay array → train) end-to-end on SLURM with trivial data before spending GPU-hours on v2f-scale.

## Setup

- Bootstrap: 4 FakeNet games on map 1 (4×4, 8 qubits)
- `cycles=1`, `num_tasks=1`, `games_per_task=2`, `training_steps=3` (per phase)
- Paths on `/home/huang651/grid_mcts2/pipelines/pipeline_smoke1/` (pre-/scratch migration)

## Result

Full chain completed autonomously in ~30 min (mostly queue time). Validated:
- `afterok` dependency gating from the selfplay array to the next train job
- A2 submit-next pattern (pretrain's trailing bash submitted the selfplay+train pair)
- Lineage propagation: cycle_01.ckpt stamped `parent_ckpt=cycle_00.ckpt`, `training_steps=6`
- Events.jsonl + state.json derived snapshot
- Selfplay worker reads a ckpt produced by train_offline.py without issue

No scientific signal from this run — it's a plumbing test. Used to de-risk v2f01.

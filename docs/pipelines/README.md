# Pipelines

Autonomous self-play ↔ training pipeline runs on Narval. A pipeline run is a
chained sequence of selfplay + training SLURM jobs; one operator command
launches the whole chain via the submit-next (A2) pattern.

Distinct from:
- `docs/experiments/` — single-decision experiment rounds tracked as markdown
- `docs/selfplay/selfplay_waves.md` — historical one-shot self-play waves (single CPU array, no training)

## Pipeline terminology

| Term | Meaning |
|---|---|
| **cycle** | One selfplay+train round inside a pipeline. Cycle 0 is pretrain-only (bootstrap data, no selfplay). Cycles 1..N each produce a new dataset dir + checkpoint. |
| **epoch** (in `--train_epochs` / `--pretrain_epochs`) | One full pass through the loaded replay buffer, *inside* a single train call. |
| **pipeline run** | Start-to-finish chain, identified by `pipeline_id`. Config frozen at kickoff. |

## Anatomy of a pipeline run

```
/scratch/huang651/grid_mcts2/pipelines/pipeline_<id>/
  config.json               # frozen settings snapshot
  pipeline_params.sh        # bash-sourceable env, read by every SLURM job
  state.json                # live snapshot (derived from events.jsonl)
  events.jsonl              # append-only event log (flock-protected)
  checkpoints/
    cycle_00.ckpt           # pretrain output
    cycle_01.ckpt ... cycle_NN.ckpt
  cycle_01/                 # selfplay dataset for cycle 1
    games/<map_class>/<map_id>/batch_*.pt
    index/<node_id>.parquet
    logs/<node_id>_*.json
    manifest.json
  cycle_02/ ... cycle_NN/
  solutions/                # optional: best plans extracted for atom-viz
  eval/                     # optional: pipeline_eval.py outputs
```

Every checkpoint stamps `run_id`, `training_steps`, `parent_ckpt`, `git_sha`,
`config_hash` for full reproducibility.

## Commands

```bash
# Launch a pipeline
python pipeline_kickoff.py \
  --pipeline_root=/scratch/huang651/grid_mcts2/pipelines \
  --pipeline_id=<short_id> \
  --bootstrap_data=<path to existing dataset dir for pretrain> \
  --cycles=<N selfplay+train rounds after pretrain> \
  --pretrain_epochs=2 --train_epochs=2 \
  --sliding_window=5 \
  --num_tasks=20 --games_per_task=1000 \
  --preset=hpc \
  --config_override=key=value ...

# Deterministic eval of any checkpoint on fresh validation maps
python pipeline_eval.py \
  --ckpt=/scratch/.../pipeline_<id>/checkpoints/cycle_NN.ckpt \
  --num_maps=30 --sim_counts=200,800,3200,10000 \
  --map_num=2 --seed_base=100000 \
  --output_dir=/scratch/.../pipeline_<id>/eval/cycle_NN \
  --save_best_json

# Check status of a running pipeline
cat /scratch/.../pipeline_<id>/state.json    # live snapshot
tail -f /scratch/.../pipeline_<id>/events.jsonl  # event stream
squeue -u huang651 | grep pipe_              # active jobs
```

## Runs

| Pipeline | Date | Bootstrap | Cycles | Key result |
|---|---|---|---|---|
| [smoke1](smoke1.md) | 2026-04-21 | 4 fake games (map 1) | 1 | Plumbing smoke test; chain works end-to-end |
| [v2f01](v2f01.md) | 2026-04-21 | wave02f_reverted (20k FakeNet, map 2 fixed) | 1 | Pretrained net hit cost **10** on fixed map 2 (prior best 12). Avg 23→17 (−27%). |
| [v3a01](v3a01.md) | 2026-04-22 → 04-23 | wave03a_rnd_plancost (19k FakeNet, random 5×5) | 5 | Avg cost 22.1 → **12.99** (−41%). Inference eval: 800 sims avg=12.20, beats Kohei (15.73). See [eval/v3a01_cycle05.md](../evaluation/v3a01_cycle05.md). |
| [v3a_fast](v3a_fast.md) | 2026-04-23 | v3a01 cycle_05 ckpt | smoke only | fast_mcts Phase 1: 2.05× CPU speedup. GPU selfplay counterproductive (env is bottleneck). |
| [v4a01](v4a01.md) | planned | v4a_gumbel FakeNet wave (8×8) | 5 planned | First 8×8 pipeline. Gumbel fixed-σ + fast backend. Bootstrap data TBD. |

## Failure modes seen in practice

- **Shared inode quota (`/project/rrg-aspuru`, 500K limit across all group members)** — v2f01's first attempt crashed at ~498K inodes. Fix: moved all experiment data to `/scratch/huang651` (1M-inode personal quota). See also [project_storage_layout.md](../../memory/...) in memory.
- **`afterok` on array jobs is all-or-nothing** — any single task failure cancels the whole array and blocks the downstream train job. Plan doc mentions this; we haven't yet switched to `afternotok`+recovery logic. For now, one failed task kills the pipeline.

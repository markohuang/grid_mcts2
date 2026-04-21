# Autonomous selfplay + training pipeline — task document

Scope: make the end-to-end selfplay ↔ actor-critic training loop run autonomously on Narval. A **pipeline run** is a chain of N epochs, where each epoch is one self-play wave (many CPU jobs) followed by one training pass (one GPU job). The pipeline fires the next selfplay epoch on training completion, and fires training on selfplay completion. Distinct from the single-variant waves tracked in `selfplay_waves.md` and `selfplay_v3.md`.

This document is the plan — not the implementation. It lists what exists, what's missing, the assumptions we'd be making, and the decisions we need before I touch code.

---

## What we want, precisely

A pipeline run is an orchestrated sequence:

```
epoch 0:  selfplay (20 CPU jobs, fake or random-init net)
          ↓ on all jobs completing
          training (1 GPU job, reads epoch 0 dataset, publishes ckpt_v0)
          ↓ on training success
epoch 1:  selfplay (20 CPU jobs, uses ckpt_v0)
          ↓ ...
          training (publishes ckpt_v1, reads epoch ≤1 dataset)
          ↓ ...
epoch N:  ...
          ↓ on termination condition
          pipeline done.
```

Operator submits one command at the start; the pipeline runs to completion (or failure) without further intervention. A `pipeline_run_id` propagates through every job, every dataset shard, every checkpoint.

---

## What exists today (inventory)

### Already in place

1. **Per-game lineage in the parquet index** (`data.py` INDEX_SCHEMA): `weight_run_id`, `weight_gen` (training_steps of producing ckpt), `weight_file`, `weight_git_sha`, `selfplay_git_sha`, `slurm_job_id`, `slurm_array_task_id`. So once training-run-id flows in, lineage is already observable at query time.
2. **Checkpoint lineage** (`trainer.save_checkpoint`): `{run_id, parent_run, epoch, training_steps, git_sha, git_dirty, config_hash}` stamped into every ckpt file. Required field `training_steps` enforces pre-lineage ckpts are rejected on load.
3. **Slurm self-play** (`selfplay_worker.py`, `slurm/selfplay_v*.sh`): takes `--weights_path` (for warm-start), writes to parquet-indexed dataset, populates lineage columns.
4. **In-process trainer** (`trainer.py`): `load_dataset_into_buffer(dataset_dir, env_config, td_steps, buffer, ...)` loads games from a parquet dir; `fit()` runs N gradient steps; `save_checkpoint(path, run_id=..., epoch=...)` writes to disk.
5. **Single-run registry**: `outputs/run_registry.jsonl` — one line per completed `main.py` run, written by `experiment.append_to_registry`. Captures config snapshot + final metrics.

### Not yet built

6. **`train_offline.py`** — standalone training entry point that reads a `dataset_dir`, loads existing checkpoint, trains, saves new checkpoint. `trainer.py` has the building blocks but no CLI wrapper.
7. **Slurm train job script** — `slurm/train_offline.sh` or similar.
8. **Pipeline orchestration** — any mechanism to chain selfplay → train → selfplay across epochs.
9. **Pipeline-run registry** — distinct from wave/experiment registry. Tracks orchestration state, not just terminal summary.
10. **Termination / failure handling** — detection of convergence, job failure, wall-time, divergence.

---

## Assumptions — please push back on any before we design further

### Hard assumptions

1. **Synchronous pipeline only** for now. One epoch's selfplay must complete before its training starts; one epoch's training must complete before next selfplay. No overlap. (lc0-style async is deferred; config_hpc.py doesn't yet support it anyway.)
2. **Slurm orchestrates, not a long-lived Python driver.** Each epoch's end submits the next with `sbatch`. No persistent driver process. Keeps the pipeline robust to login-node restarts and doesn't require an allocation for the orchestrator.
3. **Fixed epoch shape per pipeline run.** N_selfplay_jobs, games/job, sim count, training steps, etc. set at pipeline-start time and held constant for all epochs of that run. Changing them mid-run = new pipeline run.
4. **pipeline_run_id is the same as ckpt `run_id` is the same as every worker's `weight_run_id`.** One identity flows through everything. Populated at pipeline start (say `sha1(timestamp+host)[:8]`).
5. **Each epoch produces a NEW dataset dir** (`datasets/pipeline_<runid>/epoch_NN/`) and a NEW ckpt file (`checkpoints/pipeline_<runid>/epoch_NN.pt`). Previous epochs remain on disk. Trainer decides which to read.
6. **Datasets from prior epochs are valid training inputs** — replay buffer policy decides whether to include them. This is the "sliding window" question (v2 discussion: AlphaDev used 1M-game sliding window).

### Soft assumptions (worth flagging)

7. **No partial failure recovery inside a pipeline run.** If one selfplay slurm task fails, the epoch has one fewer shard; the training job proceeds on the games it has. (If 1/20 jobs fails, we lose 5% of that epoch's data — acceptable.) If training job fails, the whole pipeline halts with an operator-visible error.
8. **Configs are pinned at start of pipeline**, stored in the registry entry. Changing `config.py` mid-run doesn't affect the running pipeline (because workers read config from their per-job slurm script, which was generated at pipeline-start).
9. **Termination is driven by max epochs only** for v1 of this pipeline. Later: cost-plateau detection, value-loss convergence, wall-time cap.

---

## Design decisions that need a call before implementation

### A. How does job-dependency chaining work?

Two main options:

**Option A1: `sbatch --dependency=afterok:<prev_job_id>` chains submitted at pipeline start.**
- Operator runs `python orchestrate_pipeline.py --epochs=10` once, which submits 20 epochs worth of selfplay jobs and 10 training jobs, each dependent on the prior.
- Pro: all dependencies declared upfront; slurm's scheduler handles the whole chain.
- Con: if you want to change something at epoch 5 you have to cancel the remaining chain.
- Con: slurm has a max dependency depth (~100 on Narval I believe, need to verify).

**Option A2: "Submit-next" pattern — each job submits its successor at the end.**
- Last thing selfplay array task does (task 0 as leader) is `sbatch --dependency=afterok:<array_id>` the training job.
- Last thing training job does is `sbatch` the next selfplay array.
- Pro: state lives in what's currently running; easy to pause by killing pending jobs.
- Pro: each job can read the pipeline registry to decide "am I epoch N, what's next".
- Con: if a job fails to submit its successor silently, chain breaks. Need defensive logging.

**My lean:** A2 is more robust to long chains and easier to pause/modify mid-run, but A1 is simpler to reason about. Both are workable. **Need your call.**

### B. Where does the pipeline registry live and what shape does it take?

Options:

**B1. JSONL append log, `outputs/pipelines/<runid>/registry.jsonl`**
- One entry per epoch boundary event (selfplay_launched, selfplay_complete, training_launched, training_complete, failed, etc.)
- Easy to read, easy to grep, easy to append from concurrent jobs (with flock).
- Downside: operator needs to parse it to see "what epoch are we on".

**B2. Structured per-run JSON, `outputs/pipelines/<runid>/state.json`**
- Updated atomically on each state transition.
- Schema: `{run_id, started_at, current_epoch, epochs_complete, status, base_config_path, epochs: [{epoch_id, selfplay_job_id, selfplay_completed_at, training_job_id, training_completed_at, ckpt_path, dataset_dir, cost_median, value_loss, ...}, ...]}`.
- Easier to read at a glance; supports "what's current state of pipeline X".
- Downside: concurrent write races need careful locking.

**B3. Both.** Append-log for audit trail, state.json as the "what's live now" snapshot. State.json is derived from the append log.

**My lean:** B3 is the least regret. Append log always reflects truth; state.json is for human reading. **Need your call, but not a decision-breaker — easy to switch.**

### C. Termination criteria

**C1. Fixed N epochs** set at pipeline start.
**C2. Fixed N epochs OR cost plateau** (e.g. median hasn't improved in 3 epochs).
**C3. Fixed wall-time budget** (e.g. "run for 24 hours of pipeline wallclock, however many epochs fit").

**My lean:** C1 for v1 of the pipeline — simpler, predictable, easy to audit. Add C2/C3 later once we have epoch-to-epoch cost data to define "plateau" sensibly. **Need your call.**

### D. Replay-buffer policy for training

Per-epoch training job needs to decide which prior-epoch datasets to include in its training buffer.

**D1. Current epoch only.** Simplest. Matches older AlphaDev-ish "fresh data every step" but may lack signal.
**D2. Sliding window of last K epochs.** AlphaDev's 1M-game window analog. `K` to be tuned; start with K=3 or K=5.
**D3. All epochs from this pipeline run.** Biggest buffer; oldest data is stale relative to current net.

**My lean:** D2 with K=3 or K=5 as a starting point. Close enough to AlphaDev's sliding-window spirit. **Need your call.**

### E. What goes in each epoch's config?

Each selfplay and training job needs to read a config. Options:

**E1. One config file per pipeline run, frozen at start.** `pipelines/<runid>/config.json`. All epochs read the same config.
**E2. Per-epoch config files** auto-derived from a template, allowing e.g. ramped sim counts.
**E3. Static config from `config.py` at submission time** (what we do now for single waves).

**My lean:** E1. Freezes the pipeline behavior; mid-pipeline config changes require starting a new pipeline. **Need your call.**

---

## Open questions (I don't have answers yet, just flagging)

- **Does the `dataset_dir` per epoch interact cleanly with training buffer loading?** Current `load_dataset_into_buffer` takes one `dataset_dir`. For a sliding window of K epochs, we'd either (a) need it to take a list, (b) load K times and concatenate, or (c) consolidate to a single logical directory via symlinks or a manifest file. All three work; need to pick one.

- **Does `config.experiment.parent_run` get populated with "previous epoch's run_id" or with "prior_pipeline_run_id"?** These are two different lineage types — epoch-to-epoch continuity inside a pipeline, vs. pipeline-to-pipeline warm-start. Probably need two fields: `parent_ckpt` (prev epoch inside this pipeline) and `parent_pipeline_run` (prior pipeline this warm-started from).

- **Slurm dependency chain limits on Narval.** Need to verify whether 10+ `afterok` dependencies chained deep is fine or whether we'll hit a limit. (lc0-scale pipelines run indefinitely async; we're proposing short finite chains, so probably fine, but worth confirming with a small test.)

- **How do we cancel a running pipeline cleanly?** `scancel <job_id>` on the currently-running job stops that job, but pending jobs with `--dependency` persist in the queue. Need an `orchestrate_cancel.py` that finds all pending jobs with the matching `run_id` in their name and cancels them too. Small script, but needs to exist.

- **Logging.** Per-epoch slurm logs land in `slurm/logs/`. For a pipeline with 10 epochs × 20 workers × 2 jobs/epoch = ~220 files. Worth either symlinking them into `pipelines/<runid>/logs/` for easy navigation, or just leaving them in the flat dir and using job names to find them.

---

## Implementation order (if green-lit — rough sizing)

Each item is independently useful and testable. Ordered so each step unblocks the next without requiring anything downstream.

### 1. `train_offline.py` — single-epoch training entry point (~2 hours)

CLI: `python train_offline.py --dataset_dirs <dir1> [--dataset_dirs <dir2> ...] --load_ckpt <path> --save_ckpt <path> --training_steps <N> --run_id <id> --epoch <N>`. Loads dataset(s), constructs `AlphaAtomsTrainer`, loads prev ckpt if provided, runs `fit()` for N steps, saves stamped ckpt. No pipeline logic inside. Testable standalone.

### 2. `slurm/train_offline.sh` — GPU training job (~1 hour)

Reads env vars: `PIPELINE_RUN_ID`, `EPOCH`, `DATASET_DIRS`, `LOAD_CKPT_PATH`, `SAVE_CKPT_PATH`, `TRAINING_STEPS`. Invokes `train_offline.py`. Separate from pipeline orchestration — testable on its own with a completed wave's dataset.

### 3. Pipeline orchestrator + registry (~4-6 hours)

`orchestrate_pipeline.py`: takes `--config config.json --epochs N --run_id id --base_ckpt path`. Creates `outputs/pipelines/<runid>/` directory, writes state.json + append log, submits epoch-0 selfplay with dependency-submit of epoch-0 training which dependency-submits epoch-1 selfplay, etc.

`outputs/pipelines/<runid>/` structure:
```
state.json                      # live state
events.jsonl                    # append log
config.json                     # frozen pipeline config
epoch_00/
  selfplay_job_ids.txt
  dataset_dir -> /project/.../pipelines/<runid>/epoch_00/  (symlink)
  ckpt_v0.pt
epoch_01/
  ...
```

Depends on decisions A/B/C/D/E above.

### 4. Operator tooling (~1-2 hours)

`pipeline_status.py`: read state.json, print progress.
`pipeline_cancel.py`: find and cancel pending jobs for a run_id.
`pipeline_list.py`: list all pipeline runs with status.

### 5. Smoke test + minimal first pipeline run (~1 hour)

Fixed map, FakeNet (so step 1 doesn't actually train — just exercises load/save roundtrip), 3 epochs. Validates plumbing end-to-end.

**Total:** rough 8-12 hours spread across 3-5 sessions, each session ending with a testable deliverable.

---

## What's deferred (explicitly out of scope)

- Async/continuous pipeline (lc0-style)
- Auto-scaling of epoch size based on convergence metrics
- Multi-objective optimization (e.g. cost AND diversity targets per epoch)
- Checkpoint averaging / EMA between epochs beyond what's in the current network
- Hyperparameter search across pipeline runs
- Distributed (multi-node) training — we're single-GPU for training, multi-node only for selfplay

---

## Before I start implementing, I need from you

1. **Decisions A through E.** If you want, we can do this in one pass — I'll read your answers and start step 1.
2. **Name / location.** Is `docs/selfplay/autonomous_pipeline.md` the right home? (This isn't only selfplay — it's the full pipeline. Maybe `docs/pipelines/autonomous.md` is more honest. Either works.)
3. **Registry separation confirmation.** My read: the new pipeline registry (`outputs/pipelines/<runid>/state.json`) is independent of `outputs/run_registry.jsonl` (which would be left for single-run main.py invocations) and independent of wave-tracker docs like `selfplay_v2.md` (which track human-curated experiments). Three distinct concerns; three distinct registries. Correct?
4. **Whether v3 needs to land before this, or whether this pipeline is built against v2f/v2i/v2j data.** My lean: pipeline v1 is built and validated against single-map data (v2 datasets) because that's what exists. v3's random-map generation becomes a config knob for pipeline runs, not a prerequisite. But I could be wrong — if you want v3 as prerequisite, pipeline work blocks on the random_board port.

Once those four are nailed down, I'll execute step 1 and we iterate.

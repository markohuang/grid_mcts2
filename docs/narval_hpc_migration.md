# HPC Migration Plan: Narval (Digital Alliance of Canada)

## Context

**Bottleneck:** MCTS simulation count (`num_simulations`) is the primary compute bottleneck. Each simulation is a sequential CPU-bound tree traversal. The current training loop runs self-play on CPU workers via `ProcessPoolExecutor` (forkserver), which works locally but doesn't scale beyond a single node.

**Goal:** Leverage Narval's A100 GPUs for training and many CPU cores for parallel self-play, enabling 10–100× more MCTS simulations per epoch.

---

## Narval Hardware Summary

| Resource | Spec |
|---|---|
| GPU | NVIDIA A100 40 GB HBM2e |
| CPUs/node | Up to 64 (job-visible) |
| RAM/node | Up to 249 GB |
| Max job duration | 7 days (168 hours) |
| Interconnect | 100 Gb/s Ethernet |
| Storage | `/home` (small), `/scratch` (large, purged), `/project` (persistent) |
| Local job storage | `$SLURM_TMPDIR` (fast NVMe, job-lifetime only) |

**Self-play math:** With `N` CPU workers each running 1 game, and branching factor ~12–15, 200 simulations per move, and ~30–40 moves per episode, each game takes roughly `200 * 30 * (env_clone + net_inference)` operations. On Narval you can realistically run 32–48 workers simultaneously per node, giving ~32–48× speedup in self-play throughput vs. 1 sequential game.

---

## Phase 0: Account & Environment Setup

**Checkpoint:** Can `ssh narval.alliancecan.ca` and run a Python import successfully.

### Steps

1. **Get DRAC account** — apply at `ccdb.alliancecan.ca`, join your PI's allocation group.

2. **Transfer code**
   ```bash
   rsync -avz --exclude='outputs/' --exclude='__pycache__/' \
     ~/grid_mcts2_prior_learning/ \
     narval:~/grid_mcts2_prior_learning/
   ```

3. **Set up Python environment** — Narval has no internet access from compute nodes; install from login node.
   ```bash
   # On Narval login node
   module load StdEnv/2023 python/3.11 cuda/12.2

   # Create a virtualenv in /project (persists across jobs)
   python -m venv ~/projects/def-<pi>/shared/venvs/grid_mcts2
   source ~/projects/def-<pi>/shared/venvs/grid_mcts2/bin/activate

   # Install from Alliance pre-built wheels (fast, no compilation)
   pip install --no-index torch torchvision
   pip install lightning tensordict torchrl einops numpy ml_collections absl-py
   ```
   > **Note:** Check `pip install --no-index --find-links ~/.local/lib` for available wheel versions, or use `avail_wheels torch` on the login node to list compatible builds.

4. **Smoke test** (run on login node, short)
   ```bash
   python main.py --config.use_fake=True --config.training.epochs=2 \
     --config.training.num_parallel_games=2
   ```

5. **Storage layout**
   ```
   ~/scratch/grid_mcts2/outputs/    ← active run outputs (fast I/O)
   ~/projects/def-<pi>/grid_mcts2/checkpoints/  ← persistent checkpoints
   ~/projects/def-<pi>/grid_mcts2/venvs/        ← virtualenv
   ```

---

## Phase 1: Single-Node GPU Job (Direct Port)

**Checkpoint:** A full training run completes on 1 A100 with correct metrics logged, checkpoints saved to `/scratch`.

### Job script: `slurm/train_single.sh`

```bash
#!/bin/bash
#SBATCH --job-name=grid_mcts2
#SBATCH --account=def-<pi>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8          # self-play workers + training thread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%j.out

module load StdEnv/2023 python/3.11 cuda/12.2
source ~/projects/def-<pi>/grid_mcts2/venvs/grid_mcts2/bin/activate

# Copy outputs dir to fast local storage, work there, copy back at end
mkdir -p $SLURM_TMPDIR/outputs
# If resuming, copy previous checkpoint:
# cp ~/scratch/grid_mcts2/checkpoints/latest.ckpt $SLURM_TMPDIR/

python main.py \
  --config.map_num=2 \
  --config.training.epochs=50 \
  --config.training.num_selfplay=20 \
  --config.training.num_parallel_games=6 \
  --config.mcts.num_simulations=200 \
  --config.training.batch_size=256 \
  --config.training.training_steps=200 \
  --config.training.accelerator=gpu \
  --config.training.devices=1 \
  --config.experiment.output_dir=$SLURM_TMPDIR/outputs

# Copy results back to scratch
cp -r $SLURM_TMPDIR/outputs/* ~/scratch/grid_mcts2/outputs/
```

### What to verify
- `nvidia-smi` during job: GPU utilization >50% during training steps
- `metrics.jsonl`: `selfplay_time` and `train_time` look reasonable
- Checkpoint appears in `$SLURM_TMPDIR/outputs/<run_id>/checkpoints/`

---

## Phase 2: CPU Self-Play Scaling

**Checkpoint:** Self-play throughput scales near-linearly with `num_parallel_games` up to ~24 workers; selfplay_time / train_time ratio identified.

### Key insight
MCTS self-play is embarrassingly parallel at the game level — each game is fully independent. The current `ProcessPoolExecutor` (forkserver) implementation is already correct. On Narval, you can allocate 32–48 CPU cores and run that many games in parallel.

### How to tune

1. **Profile time split** — in `metrics.jsonl`, check `selfplay_time` vs `train_time` per epoch.
   - If `selfplay_time >> train_time`: add more CPU workers or reduce simulations per game
   - If `train_time >> selfplay_time`: increase `training_steps` or `batch_size`

2. **Scale workers** — update job script:
   ```bash
   #SBATCH --cpus-per-task=32
   ```
   ```bash
   python main.py \
     --config.training.num_parallel_games=28 \    # leave ~4 cores for OS/training
     --config.mcts.num_simulations=400 \
     --config.training.num_selfplay=50
   ```

3. **Batched network inference** — current implementation calls `network.inference()` once per simulation leaf node (serial). This is the main per-worker bottleneck. At high simulation counts, the network overhead per simulation becomes dominant.
   - At `num_simulations=50`: env clone dominates
   - At `num_simulations=500+`: `network.inference()` dominates
   - **Future optimization (Phase 4):** batch leaf evaluations across concurrent simulations (virtual loss + batch inference)

4. **SLURM test job** to profile before committing to long runs:
   ```bash
   #SBATCH --time=1:00:00
   # Run 3 epochs with varying num_parallel_games: 1, 8, 16, 32
   # Read selfplay_time from metrics.jsonl
   ```

---

## Phase 3: Long Training via Job Chaining

**Checkpoint:** A 200-epoch run can survive SLURM's 7-day wall-time limit by chaining dependent jobs that resume from checkpoint.

### Problem
SLURM max job time = 7 days. A long curriculum run (200+ epochs) may exceed this.

### Solution: `--dependency=afterok`

The existing `load_checkpoint` config flag + `curriculum_initial_phase` already support resumption. Wire this into a chain:

**Submit script: `slurm/submit_chain.sh`**
```bash
#!/bin/bash
# Usage: bash submit_chain.sh <num_jobs> <epochs_per_job> <checkpoint_dir>
NUM_JOBS=${1:-3}
EPOCHS=${2:-50}
CKPT_DIR=${3:-~/scratch/grid_mcts2/checkpoints}
CURRICULUM_MAPS=10

JID=$(sbatch --parsable slurm/train_resume.sh "" 0 $EPOCHS $CURRICULUM_MAPS)
echo "Job 1: $JID"

for i in $(seq 2 $NUM_JOBS); do
    JID=$(sbatch --parsable --dependency=afterok:$JID \
        slurm/train_resume.sh $CKPT_DIR/job_${i-1}_final.ckpt \
        $((($i-1)*$EPOCHS)) $EPOCHS $CURRICULUM_MAPS)
    echo "Job $i: $JID (depends on previous)"
done
```

**Resume-aware job script: `slurm/train_resume.sh`**
```bash
#!/bin/bash
#SBATCH --job-name=grid_mcts2_resume
#SBATCH --account=def-<pi>
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%j.out

CKPT_PATH=$1          # empty string = fresh start
CURRICULUM_START=$2   # curriculum_initial_phase
EPOCHS=$3
CURRICULUM_MAPS=$4

module load StdEnv/2023 python/3.11 cuda/12.2
source ~/projects/def-<pi>/grid_mcts2/venvs/grid_mcts2/bin/activate

LOAD_FLAG=""
if [ -n "$CKPT_PATH" ]; then
    LOAD_FLAG="--config.experiment.load_checkpoint=$CKPT_PATH"
fi

python main.py \
  --config.map_num=2 \
  --config.training.epochs=$EPOCHS \
  --config.training.num_parallel_games=28 \
  --config.mcts.num_simulations=400 \
  --config.training.batch_size=256 \
  --config.training.training_steps=200 \
  --config.training.accelerator=gpu \
  --config.experiment.output_dir=~/scratch/grid_mcts2/outputs \
  --config.experiment.curriculum_maps=$CURRICULUM_MAPS \
  --config.experiment.curriculum_initial_phase=$CURRICULUM_START \
  $LOAD_FLAG

# Copy final checkpoint to persistent project storage
RUNID=$(ls -t ~/scratch/grid_mcts2/outputs | head -1)
cp ~/scratch/grid_mcts2/outputs/$RUNID/checkpoints/final.ckpt \
   ~/projects/def-<pi>/grid_mcts2/checkpoints/job_${SLURM_JOB_ID}_final.ckpt
```

> **Note:** `curriculum_initial_phase` tells `main.py` how many curriculum maps to pre-populate before epoch 0. This is critical for resumption — otherwise the map pool restarts from scratch.

---

## Phase 4: Batched MCTS Inference (Major Throughput Improvement)

**Checkpoint:** `network.inference()` is batched across N concurrent MCTS simulations; single-game simulation throughput increases by ~N×.

### Context
Currently `run_mcts()` calls `network.inference()` once per simulation, serially. On a GPU, the cost of a small batch vs. a single inference is nearly identical — so batching M leaf evaluations costs ~1 inference instead of M. This is the "virtual loss + batch evaluation" pattern used in production AlphaZero systems.

### Design sketch

This requires restructuring `run_mcts()` to run simulations in a coroutine-style (or breadth-first) pattern:

```
Instead of:
  for sim in range(N):
    traverse → leaf → network_call(1) → backprop

Do:
  batch_leaves = []
  for sim in range(N):
    traverse to leaf (with virtual loss) → collect leaf state
    batch_leaves.append(leaf_state)

  network_outputs = network.inference(batch_leaves)  # single GPU call

  for sim, output in zip(sims, network_outputs):
    expand + backprop
```

**Implementation notes:**
- Virtual loss: temporarily add `-1` to a node's value during traversal so other simulations don't collapse to the same node
- This is a moderate refactor of `mcts.py:run_mcts()`
- Start with batch size = 8 or 16 (tune based on GPU utilization)
- Prerequisite: `network.inference()` must accept batched `obs_features` (already supported for training, check inference path)

**When to prioritize:** Only worth doing if profiling shows network inference is >30% of `selfplay_time`. At `num_simulations=200` on CPU with small maps, env cloning may still dominate.

---

## Phase 5: Multi-Node Self-Play (Optional, Advanced)

**Checkpoint:** Self-play workers run across multiple nodes, with a central training process that consumes games and broadcasts updated weights.

### When needed
Only if a single node (32–48 CPU cores) doesn't provide enough self-play throughput to keep the GPU saturated during training.

### Architecture options

**Option A: Embarrassingly parallel job array**
- Submit N independent jobs, each doing self-play and saving games to shared `/scratch`
- A separate training job reads from the shared buffer and periodically writes updated weights
- Simplest; no inter-node communication needed
- Limitation: stale weights (workers use checkpoint from start of job)

**Option B: Ray or similar**
- Use [Ray](https://docs.ray.io/en/latest/index.html) for distributed actor-based self-play
- Workers are Ray actors; training is a separate actor; weights broadcast via Ray's object store
- More complex setup on SLURM but well-supported (see `ray.init()` with SLURM integration)
- Alliance Canada has [Ray docs](https://docs.alliancecan.ca/wiki/Ray)

**Option C: torch.distributed + async replay**
- Not recommended here — designed for gradient synchronization, not actor-critic RL

**Recommendation:** Start with Option A (job array) as it requires zero code changes. Move to Ray only if staleness becomes a measurable training quality issue.

---

## Milestone Summary

| Phase | Goal | Key metric | Est. effort |
|---|---|---|---|
| 0 | Narval account + env | `python main.py --use_fake=True` runs | 1–2 days |
| 1 | Single GPU job | 1 full run completes, metrics logged | 1 day |
| 2 | CPU self-play scaling | selfplay_time vs train_time profiled, 28+ workers | 1 day |
| 3 | Job chaining / long runs | 200-epoch run across multiple jobs | 1 day |
| 4 | Batched MCTS inference | Single-game sim throughput profiled + improved | 3–5 days |
| 5 | Multi-node (optional) | Only if Phase 2 saturates | variable |

---

## Quick Diagnostics Cheat Sheet

```bash
# On Narval: check partitions and GPU availability
sinfo -o "%P %G %l %C"

# Check available Python/PyTorch wheels
avail_wheels torch

# Monitor your running jobs
squeue -u $USER

# Check efficiency of a completed job
seff <JOBID>

# Tail live output
tail -f slurm/logs/<JOBID>.out

# Check replay buffer throughput
# In metrics.jsonl: selfplay_time / num_selfplay = time per game
python -c "
import json
lines = open('outputs/<run_id>/metrics.jsonl').readlines()
for l in lines[-5:]:
    m = json.loads(l)
    print(f'epoch={m[\"epoch\"]} sp={m[\"selfplay_time\"]}s train={m[\"train_time\"]}s')
"
```

---

## Known Gotchas

- **No internet on compute nodes**: pre-install everything from the login node; `pip install --no-index` uses Alliance pre-cached wheels.
- **`forkserver` vs `fork`**: current code already uses `forkserver` context (`trainer.py:91`) — this is correct for CUDA-safe multiprocessing.
- **`torch.set_num_threads(1)`**: already set per worker (`trainer.py:13`) — critical to prevent CPU thrashing when running 32 workers.
- **`$SLURM_TMPDIR` is node-local**: don't write checkpoints there if you want them to persist after the job ends. Always copy to `/scratch` or `/project` in a trap or at job end.
- **Memory**: each self-play worker holds a copy of the network weights in CPU memory. With 32 workers and a ~10 MB model, this is negligible.
- **Scratch purge policy**: `/scratch` files inactive for 60 days may be deleted. Copy important checkpoints to `/project`.

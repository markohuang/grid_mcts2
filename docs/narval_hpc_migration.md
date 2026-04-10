# HPC Scale-Up Plan: Distributed Self-Play on Narval

## Big Picture

MCTS training quality scales with data volume — lc0 demonstrated that ~800 visits/move across hundreds of millions of games produces superhuman play. Our current single-machine setup generates ~20 games/epoch with 50-200 simulations/move, which is orders of magnitude below what's needed for the 5×5+ maps.

**Strategy:** Decouple self-play (CPU-intensive) from training (GPU-intensive).

1. **Phase 1 (now):** Use Narval CPU nodes purely for massive self-play data generation. Build a structured dataset on `/project`. Manually rsync to a GPU machine for training.
2. **Phase 2 (later):** Move to a cluster with both GPU and CPU nodes. Add async weight loop (workers pull latest weights, trainer consumes games continuously — the lc0 pattern).

```
Phase 1: Narval CPU → dataset on /project → rsync → GPU train elsewhere
Phase 2: GPU+CPU cluster, async weight updates, continuous pipeline
```

---

## Narval Hardware (CPU Allocation)

| Resource | Spec |
|---|---|
| CPUs/node | Up to 64 cores |
| RAM/node | Up to 249 GB |
| Max job duration | 7 days (168 hours) |
| Interconnect | 100 Gb/s Ethernet |
| `/home` | Small quota, **read-only on compute nodes** |
| `/scratch` | Large, read/write, purged after 60 days inactive |
| `/project` | Persistent, read/write, shared across group |
| `$SLURM_TMPDIR` | Fast node-local NVMe, job-lifetime only |

---

## Target Scale

**200k games at 5×5 with `num_simulations=1000`:**
- Episode length ≈ 30 steps (sum of relevant atoms across 3 layers)
- Per game: 30 steps × 1000 sims × 1 network inference/sim = 30k inferences
- On 1 CPU core, ~100 inferences/sec (small MLP) → ~300 sec/game → ~5 min/game
- On 100 nodes × 60 cores = 6000 workers → ~33 games/sec → **200k games in ~1.7 hours**
- Dataset size: 200k games × ~4 KB/game (raw format) ≈ **800 MB**

For context, lc0 generates billions of positions. 200k games × 30 positions = 6M positions — a modest start that validates the infrastructure before scaling further.

---

## Dataset Format

### Directory structure

```
/project/<user>/grid_mcts2/datasets/<dataset_name>/
├── manifest.json                          # dataset metadata, config snapshot
├── maps/
│   └── <map_id>.json                      # full map spec per unique map
└── games/
    └── <map_class>/                       # e.g., 5x5_12q_04g_03l
        └── <map_id>/                      # e.g., a3f7c2e1
            ├── batch_node01_000001.pt     # list of compact game dicts
            ├── batch_node01_000002.pt
            └── batch_node37_000042.pt
```

### Map taxonomy

**map_class** = `{H}x{W}_{Q:02d}q_{G:02d}g_{L:02d}l`
- Encodes board dimensions, qubit count, max gates/layer, number of layers
- Example: `5x5_12q_04g_03l` = 5×5 board, 12 qubits, 4 gates/layer, 3 layers
- Used for filtering: "give me all 5×5 games" or "train a specialist on this map class"

**map_id** = `md5(canonical_json(sorted_atom_map, sorted_tasks))[:8]`
- Unique per specific map instance (atom positions + gate assignments)
- Two maps in the same class but with different atom layouts get different IDs
- Used for: specialist training on a single map, or tracking per-map performance

### Game serialization (compact)

Each game is stored as a plain dict (not pre-computed TensorDict):

```python
{
    'map_id': 'a3f7c2e1',
    'map_class': '5x5_12q_04g_03l',
    'history': [3, 7, 12, ...],           # action indices
    'rewards': [0.0, -0.5, ...],          # per-step rewards
    'child_visits': [[0.1, 0.0, ...], ...],  # MCTS visit distributions
    'root_values': [1.2, 0.8, ...],       # MCTS root values
    'latency_reward': -0.3,               # terminal latency signal
    'num_simulations': 1000,              # MCTS sims used
    'weight_gen': 0,                      # which weight generation produced this
}
```

**~4 KB/game** vs ~300 KB for pre-computed features. The trainer replays actions through the env to reconstruct features on-the-fly — this is fast (no MCTS, just `env.step()` calls) and decouples the data format from the feature representation.

### Batch files

Each `.pt` file contains a list of game dicts (one worker's output batch, typically 60 games). Using `torch.save`/`torch.load` for simplicity. Files are written atomically (write to `.tmp`, then `os.rename`).

---

## Architecture

### Self-play workers (`selfplay_worker.py`)

Each SLURM task runs on one node with 60 CPU cores:

```
1. Load config + optional network weights
2. Spawn 60 ProcessPoolExecutor workers
3. Each worker plays 1 game (MCTS with num_simulations=1000)
4. Collect finished games into a batch
5. Save batch to /project/.../games/pending/<node_id>_batch_<seq>.pt
6. Repeat until wall-time limit or target game count reached
```

Workers are stateless — they read weights once at startup (Phase 1) or poll for updates (Phase 2). No inter-node communication.

### SLURM job array

```bash
#SBATCH --array=0-99          # 100 nodes
#SBATCH --cpus-per-task=64    # all cores
#SBATCH --mem=64G
#SBATCH --time=3:00:00
```

Each array task writes to the same dataset directory on `/project`. No race conditions because each task writes to uniquely-named files (includes `$SLURM_ARRAY_TASK_ID` in filename).

### Trainer (runs elsewhere, Phase 1)

After rsync-ing the dataset:

```python
# Load all games from dataset, filtering by map_class or map_id
buffer = load_dataset_into_buffer("datasets/run01/", filter_class="5x5_12q_04g_03l")
# Train using existing fit() loop
trainer.fit()
```

The trainer reconstructs TensorDict training samples from raw game dicts by replaying through the env. This reuses the existing `save_game()` logic in `trainer.py`.

---

## Code Changes

### Files to modify

| File | Change |
|---|---|
| `neutral_atoms/game.py` | Add `to_dict()` / `from_dict()` for compact serialization |
| `neutral_atoms/config.py` | Add `map_class(map_data)` and `map_id(map_data)` utility functions |
| `neutral_atoms/trainer.py` | Extract `game_to_tensordict(game, config)` from `save_game()`; add `load_dataset_into_buffer(path, filter_fn)` |
| `neutral_atoms/experiment.py` | `fcntl.flock()` around `append_to_registry()` |

### Files to create

| File | Purpose |
|---|---|
| `neutral_atoms/data.py` | Dataset I/O: `atomic_save_batch()`, `scan_dataset()`, `dataset_stats()`, manifest management |
| `selfplay_worker.py` | Entry point for CPU self-play on Narval |
| `slurm/selfplay.sh` | SLURM job array script |

### Files untouched

`mcts.py`, `env.py`, `board.py`, `moves.py`, `rewards.py`, `network.py`, `main.py` — no changes. The existing synchronous `main.py` loop continues to work for local development.

---

## Phased Implementation

### Phase 0: Narval Environment

**Checkpoint:** `python main.py --config.use_fake=True --config.training.epochs=2` runs on Narval login node.

1. Get DRAC account, join PI allocation
2. Transfer code via rsync
3. Set up virtualenv on `/project` (login node has internet)
   ```bash
   module load StdEnv/2023 python/3.11
   python -m venv ~/projects/def-<pi>/venvs/grid_mcts2
   source ~/projects/def-<pi>/venvs/grid_mcts2/bin/activate
   pip install --no-index torch
   pip install lightning tensordict torchrl einops numpy ml_collections absl-py
   ```
4. Smoke test with FakeNet

### Phase 1: Self-Play Data Pipeline

**Checkpoint:** 100 SLURM array tasks each produce game batches on `/project`; `dataset_stats()` reports total games and per-map breakdowns.

1. Implement `Game.to_dict()` / `from_dict()` + roundtrip test
2. Implement `map_class()` / `map_id()` in config.py
3. Build `neutral_atoms/data.py` — atomic writes, dataset scanning
4. Build `selfplay_worker.py` — the CPU entry point
5. Write `slurm/selfplay.sh` — job array submission
6. Test locally: run worker with `--num_games=5 --num_workers=2`, verify dataset structure
7. Test on Narval: small array (4 nodes), verify files land on `/project`
8. Scale to 100 nodes, generate 200k games

### Phase 2: Train from Dataset

**Checkpoint:** Training on GPU machine using rsync'd dataset matches or exceeds locally-generated training quality.

1. `game_to_tensordict()` extraction in trainer.py
2. `load_dataset_into_buffer()` that filters by class/id
3. Verify training loop works with dataset-loaded buffer
4. rsync dataset from Narval, train on GPU machine

### Phase 3: Async Weight Loop (Future)

**Checkpoint:** Workers poll for weight updates; trainer publishes weights after each training cycle.

1. Weight versioning: `weights/gen_{N:06d}.pt` + `weights/latest.pt` symlink
2. Workers check `latest.pt` mtime, reload when changed
3. `train_async.py` — continuous trainer that polls `games/pending/`, trains, publishes
4. Move to GPU+CPU cluster

---

## Narval Gotchas

- **`/home` is read-only on compute nodes** — write outputs to `/scratch` or `/project`
- **No internet on compute nodes** — pre-install all packages from login node
- **SQLite on Lustre is unreliable** — use file-per-batch pattern instead of databases
- **Atomic writes** — always write to `.tmp` then `os.rename()` for crash safety on Lustre
- **`forkserver` context** — already used in `trainer.py:91`, correct for multiprocessing
- **`torch.set_num_threads(1)`** — already set per worker in `trainer.py:13`, prevents CPU thrashing
- **Scratch purge** — files inactive 60 days get deleted; copy important data to `/project`
- **Max 1000 jobs queued** per user — plan array sizes accordingly

## Diagnostics

```bash
# On Narval
sinfo -o "%P %l %C"               # partitions, time limits, CPU counts
squeue -u $USER                    # running/pending jobs
seff <JOBID>                       # efficiency report after completion
sacct -j <JOBID> --format=Elapsed,MaxRSS,NCPUS  # resource usage

# Dataset health
python -c "
from neutral_atoms.data import dataset_stats
dataset_stats('/project/.../datasets/run01/')
"
```

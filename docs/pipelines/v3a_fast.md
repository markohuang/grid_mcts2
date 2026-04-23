# v3a_fast — fast_mcts integration on 5×5 random boards

Experiment series building on v3a01 with `backend=fast` (`fast_mcts` Phase 1 leaf-gather +
virtual loss). Same map class (MAPS[2], random 5×5, 12 qubits, 3 layers), same bootstrap
weights (v3a01 cycle_05). Goal: reduce selfplay wall time per cycle so we can scale up
sims, cycles, or both within the same compute budget.

**Parent run:** `pipeline_v3a01` (5 cycles, classic/CPU, avg cost 22.1 → 12.99, see [v3a01.md](v3a01.md))

---

## What changed from v3a01

| Knob | v3a01 | v3a_fast |
|---|---|---|
| `mcts.backend` | `classic` | `fast` |
| `mcts.nn_batch_size` | — (ignored) | `32` |
| `mcts.virtual_loss` | — (ignored) | `1.0` |
| `selfplay_device` | `cpu` | `cpu` (smoke) / `cuda` (GPU test) |
| everything else | unchanged | unchanged |

Config defaults (`config.py`) now carry all three new fields so they're CLI-overridable
and flow through `pipeline_kickoff.py --config_override=...` → `$CONFIG_FLAGS` unchanged.

---

## v3a01 cycle_05 deterministic eval (partial, context for baseline)

Job 59778046 timed out before completing sims=10000 (no `eval_summary.json`).
Per-sim best-cost on 30 held-out validation maps (seed_base=100000):

| sims | best cost |
|---|---|
| 200 | **10** |
| 800 | **10** |
| 3200 | **10** |
| 10000 | — (timed out) |

**Key implication**: trained prior reaches cost=10 at 200 sims — ~50× sim reduction vs the
FakeNet bootstrap (avg=22.1 at 10k sims). The sims=10000 eval is deferred to after the fast
backend is validated (will be much faster with GPU).

---

## Sanity checks — must pass before trusting smoke numbers

Script: `fast_mcts/test_gpu_sanity.py`. Runs automatically at the start of the GPU smoke job.
Also runnable standalone: `python fast_mcts/test_gpu_sanity.py --ckpt <cycle_05.ckpt> --sims=1000`

| # | Check | Pass condition |
|---|---|---|
| 1 | GPU classic vs GPU fast (batch=1, vl=0): byte-identical trajectories (5 seeds) | All 5 seeds match |
| 2 | H2D fraction at batch=32, 10k sims: `total_h2d_ms / total_nn_ms` | < 20% |
| 3 | Quality parity: GPU fast batch=32 vl=1 vs GPU classic, 20 games | `\|Δmean_cost\| ≤ 1.5` |

GPU util target during selfplay: >40% sustained (logged by `nvidia-smi` every 5s in
`$DATASET_DIR/nvidia_smi.log`). If <15%, model is too small to saturate the GPU kernel —
consider larger batch or larger network before concluding GPU selfplay is not worthwhile.

---

## Smoke runs

### Smoke A — CPU fast (wallclock baseline)

**Script:** `slurm/selfplay_v3a_fast_smoke.sh`  
**Config:** `backend=fast, nn_batch_size=32, virtual_loss=1.0, selfplay_device=cpu`  
**Setup:** 62 games, 62 parallel workers, 10k sims, v3a01 cycle_05 ckpt, random 5×5  
**SLURM job:** 59789765  
**Dataset:** `/scratch/huang651/grid_mcts2/datasets/v3a_fast_smoke`

Comparison baseline is v3a01 cycle_05 at **537s/game** (classic/CPU, same ckpt).

| Metric | v3a01 c5 (classic/CPU) | Smoke A (fast/CPU) | notes |
|---|---|---|---|
| avg_game_time_s | 537 | **TBD** | |
| avg_cost | 12.99 | TBD | should match within noise |
| nn_eff_batch | — | TBD | target ≥ 28 (batch=32) |
| nn_h2d_frac | — | 0.0 (expected) | CPU: no H2D |
| speedup | 1.0× | **TBD** | Phase 1 bench CPU: 5.35× at 100 sims |

### Smoke B — GPU fast, 1 worker (per-game timing)

**Script:** `slurm/selfplay_v3a_fast_gpu_smoke.sh`  
**Config:** `backend=fast, nn_batch_size=32, virtual_loss=1.0, selfplay_device=cuda`  
**Setup:** 10 games, **1 worker** (sequential — clean per-game timing), 10k sims, same ckpt  
**SLURM job:** 59789771 (pending Priority)  
**Dataset:** `/scratch/huang651/grid_mcts2/datasets/v3a_fast_gpu_smoke`

| Metric | Smoke B (fast/GPU, 1w) | notes |
|---|---|---|
| Sanity check 1 (parity) | TBD | |
| Sanity check 2 (H2D frac) | TBD | target < 0.20 |
| Sanity check 3 (quality) | TBD | |
| avg_game_time_s | **TBD** | |
| avg_cost | TBD | |
| nn_eff_batch | TBD | |
| nn_h2d_frac | TBD | |
| GPU util % (nvidia-smi) | TBD | target > 40% |
| speedup vs v3a01 c5 (537s) | TBD | |
| speedup vs Smoke A | TBD | |

### Smoke B2 — GPU fast, 4 workers (throughput)

Same script, `NUM_WORKERS=4 NUM_GAMES=20`. Runs after Smoke B confirms parity.

4 concurrent CUDA contexts; safe on A100 (model ~few MB × 4 + ~300 MB/context overhead).

| Metric | 1 worker | 4 workers |
|---|---|---|
| avg_game_time_s | TBD | TBD |
| wall for 20 games | TBD | TBD |
| GPU util % | TBD | TBD |

---

## Decision criteria → v3a_fast pipeline

| Outcome | Action |
|---|---|
| GPU ≥ 3× faster than CPU-fast AND H2D < 20% AND quality OK | Switch pipeline to GPU node + fast backend; keep 10k sims |
| GPU 1.5–3× faster | Weigh GPU core-hour cost vs CPU; likely still worth it |
| GPU < 1.5× faster | Stay CPU; spend GPU budget on training not selfplay |
| H2D > 20% | Try batch=64 or 128; otherwise GPU doesn't help this model size |
| Quality degrades > 1.5 cost units | Lower virtual_loss to 0.5 or batch to 16 |

---

## Planned pipeline ablations (v3a_fast_01 …)

Once smoke confirms speedup, pipeline runs use v3a01 cycle_05 as bootstrap ckpt:

| Run | backend | batch | vl | device | sims | cycles | purpose |
|---|---|---|---|---|---|---|---|
| `v3a_fast_01` | fast | 1 | 0.0 | cpu | 10k | 1 | parity sanity over full cycle |
| `v3a_fast_02` | fast | 32 | 1.0 | cpu | 10k | 5 | primary CPU-fast treatment |
| `v3a_fast_03` | fast | 32 | 1.0 | cuda | 10k | 5 | primary GPU-fast treatment |
| `v3a_fast_04` | fast | 32 | 1.0 | cuda | 50k | 5 | cash GPU speedup into 5× more sims |

Kickoff template (fill in `pipeline_id` and `DEVICE`):
```bash
python pipeline_kickoff.py \
  --pipeline_root=/scratch/huang651/grid_mcts2/pipelines \
  --pipeline_id=v3a_fast_02 \
  --bootstrap_ckpt=/scratch/huang651/grid_mcts2/pipelines/pipeline_v3a01/checkpoints/cycle_05.ckpt \
  --cycles=5 \
  --pretrain_epochs=0 \
  --train_epochs=2 \
  --sliding_window=5 \
  --num_tasks=20 --games_per_task=1000 \
  --selfplay_time=02:00:00 --train_time=01:30:00 \
  --selfplay_cpus=64 --selfplay_mem=32G --train_mem=64G \
  --preset=hpc \
  --config_override=map_num=2 \
  --config_override=random_board=True \
  --config_override=mcts.num_simulations=10000 \
  --config_override=env.reward_mode=plan_cost \
  --config_override=mcts.root_dirichlet_alpha=0.1 \
  --config_override=training.buffer_size=2500000 \
  --config_override=mcts.backend=fast
```

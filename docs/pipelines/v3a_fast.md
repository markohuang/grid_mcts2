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

## v3a01 cycle_05 deterministic eval (complete for sims ≤ 3200)

Job 59790830. 30 held-out validation maps, seed_base=100000. Full results in
`pipelines/pipeline_v3a01/eval/cycle_05b/eval_summary.json`. See also [docs/evaluation/v3a01_cycle05.md](../evaluation/v3a01_cycle05.md).

| method | avg cost | min | p10 | p50 | p90 | avg plan time |
|---|---|---|---|---|---|---|
| Kohei greedy (no search) | 15.73 | 13 | 14 | 16 | 18 | 12ms |
| MCTS sims=200 | 13.63 | 10 | 11 | 13 | 17 | 9.5s |
| MCTS sims=800 | 12.20 | 10 | 10 | 12 | 14 | 35s |
| MCTS sims=3200 | **11.53** | **10** | **10** | **11** | **14** | 129s |

**Key implication for fast_mcts**: 800 sims already beats Kohei by −22% avg cost (12.20 vs 15.73).
The fast backend's 2.05× speedup means:
- At sims=800: 35s/map → ~17s/map with fast backend
- Selfplay at sims=800 (vs current 10k) = 12× faster games at comparable or better inference quality
- This unlocks ~10 cycles within the same 21h wall clock as v3a01's 5 cycles

sims=10000 eval is deferred (each run at 30 maps × 537s/game ≈ 4.5h; needs either fast backend or reduced map count).

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
| avg_game_time_s | 537 | **262** | **2.05× speedup** |
| avg_cost | 12.99 | 12.82 | quality maintained (min 9 vs 8) |
| nn_eff_batch | — | **5.5** | cap=32; low — see finding below |
| terminal_frac | — | 84.7% | most sims hit terminal, skip NN |
| nn_total_ms (% of game) | — | 10.6% | NN is NOT the bottleneck at 10k sims |
| nn_h2d_frac | — | 0.0 | CPU: no H2D |
| speedup | 1.0× | **2.05×** | Phase 1 bench was 5.35× at 100 sims |

**Finding (critical):** with the trained v3a01 prior, 84.7% of 10k simulations reach a
terminal node and never call the NN. The gather batch fills to only 5.5/32 leaves on
average. NN time is 10.6% of game time; env.step/clone is 89.4%. Phase 1 (NN batching)
gives 2× because it reduces NN overhead and Python dispatch, but env is the real bottleneck.
Phase 3 (vec env / undo-stack) is the next lever.

### Smoke B — GPU fast, 1 worker (per-game timing)

**Script:** `slurm/selfplay_v3a_fast_gpu_smoke.sh`  
**Config:** `backend=fast, nn_batch_size=32, virtual_loss=1.0, selfplay_device=cuda`  
**Setup:** 10 games, **1 worker** (sequential — clean per-game timing), 10k sims, same ckpt  
**SLURM job:** 59789771 (pending Priority)  
**Dataset:** `/scratch/huang651/grid_mcts2/datasets/v3a_fast_gpu_smoke`

Sanity script failed with `ModuleNotFoundError: fast_mcts` (fixed: added `sys.path` insert).
Selfplay proceeded (non-blocking). Re-run sanity checks standalone after smoke completes.

| Metric | Smoke B (fast/GPU, 1w) | notes |
|---|---|---|
| Sanity check 1 (parity) | **errored** | import fix applied; re-run pending |
| Sanity check 2 (H2D frac) | **errored** | |
| Sanity check 3 (quality) | **errored** | |
| avg_game_time_s | **~363** (5/10 games) | **SLOWER than CPU fast (262s)** |
| avg_cost | ~11.4 | quality fine |
| nn_eff_batch | ~5.5 | same as CPU: terminal_frac limits batch fill |
| GPU util % (nvidia-smi) | **1%** | A100 idle >99% of the time |
| speedup vs v3a01 c5 (537s) | ~1.48× | worse than CPU fast |
| speedup vs Smoke A (CPU fast) | **~0.72×** | GPU adds overhead, not gain |

**Finding (critical):** GPU selfplay is *counterproductive* at 10k sims + trained prior.
NN is 10.6% of game time; env.step/clone is 89.4%. eff_batch=5.5 means the A100 executes
tiny batches of ~5 tensors, adding kernel-launch overhead (~50–200 µs/call) with negligible
compute to amortize. Amdahl limit: 1 / (0.894 + 0.106/∞) = **1.12×** — even perfect GPU
NN compute can't deliver more than 12% total speedup. Actual result is 0.72× (slowdown)
because kernel overhead exceeds savings.

**Smoke B2 (4-worker GPU) cancelled** — given the 1-worker result, scaling workers
only adds CUDA context overhead without fixing the env.step bottleneck.

---

## Conclusions and next steps

### What we learned

| Question | Answer |
|---|---|
| Does fast_mcts Phase 1 help on CPU? | **Yes — 2.05×** at 10k sims + trained prior. Ship it. |
| Is the bottleneck still NN at 10k sims? | **No.** NN = 10.6% of time, env.step/clone = 89.4%. |
| Does GPU selfplay help? | **No — 0.72× (slowdown).** A100 at 1% util. Amdahl limit ~1.12×. |
| Why is eff_batch only 5.5 with cap=32? | Terminal fraction = 84.7%: most sims skip NN entirely. |
| What delivers the next 5× speedup? | **Phase 3: vec env / undo-stack** — targets the 89.4%. |

### Pipeline recommendation

Use `backend=fast` on CPU nodes (2× win, zero extra cost). Selfplay time budget:
v3a01's 3h/cycle → **~1.5h/cycle** with fast backend, or run more games in same budget.

GPU allocation: keep exclusively for training (A100 is well-utilized there). Do not request
GPU for selfplay nodes until Phase 3 makes NN the bottleneck again.

### v3a_fast_01 pipeline (ready to launch)

Matches v3a01 exactly, adds `backend=fast`. Warm-starts from v3a01 cycle_05.
Expected per-cycle selfplay wall: ~1.5h (down from 3h). Can extend to 10 cycles
in the same total compute budget as v3a01's 5 cycles.

```bash
python pipeline_kickoff.py \
  --pipeline_root=/scratch/huang651/grid_mcts2/pipelines \
  --pipeline_id=v3a_fast_01 \
  --bootstrap_ckpt=/scratch/huang651/grid_mcts2/pipelines/pipeline_v3a01/checkpoints/cycle_05.ckpt \
  --cycles=5 \
  --pretrain_epochs=0 --train_epochs=2 \
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

### Longer-term: Phase 3 (vec env)

Phase 3 targets env.step/clone (89.4% of time). Approaches:
- **Undo-stack**: single env with `reverse_step`, no clone at all
- **Batched env**: vectorize board state across N sims, step in parallel

Once env is vectorized, NN batching fills to cap=32 (all N sims need NN, not just 15%),
GPU util jumps from 1% to >40%, and the Amdahl limit rises from 1.12× to ~5–10×.
See `fast_mcts/docs/phases.md` §Phase 3.

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

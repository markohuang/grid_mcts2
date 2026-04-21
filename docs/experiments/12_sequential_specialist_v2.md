# Round 12: Sequential Specialist v2 — New Architecture + Deep Search

## Context

Round 10 Phase 2 ran sequential specialist training (20 maps, loading checkpoints) with the **old architecture** (MLPMixer, flat participation flags, 139K params, 50 sims). It failed: no meta-learning emerged, starting cost stayed flat at ~14 across phases, partial catastrophic forgetting.

Round 11 showed the **new architecture** (Transformer + cross-positional pair features, 298K params) can represent cost structure (supervised r=0.954) but the RL generalist still plateaued at avg=14.5. R11 identified two issues:
1. Policy head too small (p_hsize=32, only 2 attention heads × 16 dims)
2. Training regime: each game sees a fresh random map once — not enough signal per map

This round retries sequential specialist training with all advantages stacked:
- **New architecture** (transformer + cross-positional features) — can represent the cost function
- **p_hsize=64** (up from 32) — R11 finding: policy head was undersized
- **250 simulations** (up from 50) — 5× deeper search per move
- **All 4 reward modes** — each with its own first-order question

## Pre-launch Audit

### Verified correct

| Component | Status | Notes |
|---|---|---|
| Cross-positional pair encoding | ✓ | `feat[cell_of_q1, q2] = 1.0` via matmul in `get_features()` |
| Gate pair information | ✓ | Future layers encoded, completed layers omitted (implicit progress) |
| `track_plan_delta` auto-enable | ✓ | `main.py:49-51` sets `track_plan_delta=True` when `search_bonus_weight > 0` |
| Value targets (n-step returns) | ✓ | Discounted rewards + bootstrap from target network |
| Policy targets (visit counts) | ✓ | MCTS visit distribution with temperature |
| Two-hot value encoding | ✓ | Categorical distribution over 101 bins, [-20, 5] |
| Search bonus UCB integration | ✓ | Added to UCB score, does not contaminate reward or value target |
| Reward computation per mode | ✓ | All 4 modes produce correct total episode reward |

### Known limitations (accepted for this round)

| Issue | Severity | Decision |
|---|---|---|
| `current_atom_idx` not in features | Medium | Network infers progress from board state (which qubits have moved). Explicit encoding could help but is an arch change — test current setup first. |
| Latency target is global scalar | Low | Latency weight is 0.1 (tiebreaker only). Not the bottleneck. |
| MCTS bonus normalized by `cost_ub` | Low | Conservative normalization. May underweight bonus. Monitor `avg_mcts_depth` to check. |

### Config changes from R10 sequential

| Parameter | R10 value | R12 value | Why |
|---|---|---|---|
| Architecture | MLPMixer + flat flags | Transformer + cross-pos | R11: can represent cost (r=0.954) |
| p_hsize | 32 | **64** | R11: 2 heads × 16 dims insufficient for attention |
| v_hsize | 64 | 64 | Unchanged |
| num_simulations | 50 | **250** | Deeper search → better MCTS policy targets |
| epochs_per_phase | 30 | 30 | Same |
| games_per_epoch | 50 | 50 | Same (250 sims already 5× more compute per game) |
| early_stopping_patience | 15 | 15 | Same |

## Design

### Starting point

Train a fresh specialist on fixed map2 with the new architecture and 250 sims (Run 12-base). This gives us the warm-start checkpoint AND answers whether 250 sims + p_hsize=64 improves the specialist floor.

Then run 20-phase sequential training from that checkpoint for each reward mode.

### Reward mode experiments — each with a first-order question

| Run | Reward mode | Search bonus | First-order question |
|---|---|---|---|
| **12A** | layer_delta | 2.0 | Does the new arch + deep search produce meta-learning (decreasing start cost across phases) where the old arch failed? |
| **12B** | plan_cost | 0.0 | Does plan_cost's cross-layer signal help sequential transfer, or does its per-map bias hurt? |
| **12C** | layer_completion | 0.0 (td=10) | With 250 sims, can the sparse reward bootstrap from a warm checkpoint even though it failed from scratch? |
| **12D** | remaining_cost | 0.0 | Does the simplest dense reward (−cost/episode_length per step) transfer better than plan_cost's delta formulation? |

**Why each mode matters:**

- **12A (layer_delta + bonus)**: The best-performing config. The question is whether sequential training with better arch/search produces the meta-learning that R10 couldn't.
- **12B (plan_cost)**: Plan_cost gives direct cross-layer reward (no bonus needed). But V_θ learns a shifted target (`do_nothing - remaining_cost`), which varies per map. Does this shift help or hurt transfer between maps?
- **12C (layer_completion)**: Sparse reward failed from scratch (R07, R11). But from a warm checkpoint with 250 sims, the value network already has useful estimates. Can sparse reward fine-tune per-map without catastrophic forgetting? (Sparse reward = less map-specific overfitting.)
- **12D (remaining_cost)**: `−remaining_cost / episode_length` at every step. Unlike plan_cost, this doesn't compute deltas — it directly reveals the absolute remaining cost. Simpler gradient signal, but might cause the "telescoping" issue where total reward is constant.

### Success criteria

| Metric | Success signal | Failure signal |
|---|---|---|
| Starting avg_cost trend | Decreases over phases 1→20 (e.g., 15 → 12) | Stays flat at 14+ |
| Convergence speed | Epochs to cost ≤ 12 decreases over phases | Stays at 15+ epochs |
| Map2 retention (eval_map0) | Stays ≤ 12 | Degrades to 15+ |
| Generalist probe (after phase 20) | avg < 14 on random maps | avg ≥ 15 |

### Timing estimate

- 250 sims per move, ~24 moves per game → ~6000 sims per game
- 50 games per epoch → 300K sims per epoch
- At ~500 sims/sec (CPU): ~600s per epoch (~10 min)
- 30 epochs per phase: ~5 hours per phase
- 20 phases: ~100 hours per reward mode
- 4 reward modes: ~400 hours total (sequential, or ~100h on 4 parallel GPUs)

The base specialist run (12-base) is ~10 hours.

## Commands

### 12-base: Fresh specialist with new arch + 250 sims

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 \
  --config.training.epochs=200 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.num_simulations=250 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.network.p_hsize=64 \
  --config.experiment.early_stopping_patience=100 \
  --config.experiment.study=round12_sequential_v2 \
  --config.experiment.hypothesis=base_specialist \
  --config.experiment.variant=fixed_250s_p64 \
  --config.experiment.tags=specialist,base,250sims,p64 \
  --config.experiment.notes="Fresh specialist: new arch, p_hsize=64, 250 sims, layer_delta+bonus"
```

### Sequential training script

```bash
#!/bin/bash
# Round 12: Sequential Specialist v2
# Usage: ./run_r12_sequential.sh <reward_mode> <search_bonus> <extra_flags> <run_label>
# Example: ./run_r12_sequential.sh layer_delta 2.0 "" 12A

set -e
PYTHON="../grid_mcts2/.venv/bin/python"
REWARD_MODE=${1:-layer_delta}
SEARCH_BONUS=${2:-0.0}
EXTRA_FLAGS=${3:-""}
RUN_LABEL=${4:-12X}

# Starting checkpoint from 12-base
CKPT="outputs/<12-base-run-id>/checkpoints/final.ckpt"

NUM_PHASES=20
EPOCHS_PER_PHASE=30
GAMES_PER_EPOCH=50

COMMON="
  --config.map_num=2
  --config.random_board=True
  --config.training.epochs=${EPOCHS_PER_PHASE}
  --config.training.num_selfplay=${GAMES_PER_EPOCH}
  --config.training.batch_size=128
  --config.training.seed=42
  --config.mcts.num_simulations=250
  --config.mcts.root_dirichlet_alpha=0.3
  --config.training.data_augmentation=True
  --config.env.reward_mode=${REWARD_MODE}
  --config.mcts.plan_cost_search_bonus_weight=${SEARCH_BONUS}
  --config.network.p_hsize=64
  --config.experiment.early_stopping_patience=15
  --config.experiment.study=round12_sequential_v2
  ${EXTRA_FLAGS}
"

echo "=== R12 Sequential: ${RUN_LABEL} (${REWARD_MODE}, bonus=${SEARCH_BONUS}) ==="

for phase in $(seq 1 $NUM_PHASES); do
    MAP_SEED=$((phase * 100))
    echo "--- Phase $phase/$NUM_PHASES (seed=$MAP_SEED) ---"

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    $PYTHON main.py \
      $COMMON \
      --config.random_board_seed=$MAP_SEED \
      --config.experiment.load_checkpoint=$CKPT \
      --config.experiment.hypothesis=${RUN_LABEL}_phase${phase} \
      --config.experiment.variant=phase${phase}_seed${MAP_SEED} \
      --config.experiment.tags=sequential,specialist,${RUN_LABEL},phase${phase} \
      --config.experiment.notes="R12 ${RUN_LABEL} phase $phase: ${REWARD_MODE}, bonus=${SEARCH_BONUS}, seed=$MAP_SEED"

    LATEST_RUN=$(ls -td outputs/*/ | head -1)
    CKPT="${LATEST_RUN}checkpoints/final.ckpt"

    tail -1 "${LATEST_RUN}metrics.jsonl" | python3 -c "
import json, sys
d = json.loads(sys.stdin.readline())
print(f'  ep{d[\"epoch\"]}, avg={d[\"avg_cost\"]}, best={d[\"best_cost\"]}, eval0={d.get(\"eval_map0_best_so_far\",\"?\")}, ent={d.get(\"entropy_layer0\",0):.2f}')
"
done

echo "=== ${RUN_LABEL} complete ==="
```

### Launch commands (after 12-base completes)

```bash
# 12A: layer_delta + search bonus (best config)
./run_r12_sequential.sh layer_delta 2.0 "" 12A

# 12B: plan_cost (cross-layer reward, no bonus)
./run_r12_sequential.sh plan_cost 0.0 "" 12B

# 12C: layer_completion (sparse reward, td=10)
./run_r12_sequential.sh layer_completion 0.0 "--config.training.td_steps=10" 12C

# 12D: remaining_cost (simplest dense reward)
./run_r12_sequential.sh remaining_cost 0.0 "" 12D
```

### Generalist probe (after all phases complete)

For each completed sequential run, evaluate the final checkpoint on fully random maps:

```bash
# Replace CKPT with the phase-20 checkpoint from each run
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
../grid_mcts2/.venv/bin/python main.py \
  --config.map_num=2 --config.random_board=True \
  --config.training.epochs=100 --config.training.num_selfplay=50 \
  --config.training.batch_size=128 --config.training.seed=42 \
  --config.mcts.num_simulations=250 \
  --config.mcts.root_dirichlet_alpha=0.3 \
  --config.training.data_augmentation=True \
  --config.env.reward_mode=layer_delta \
  --config.mcts.plan_cost_search_bonus_weight=2.0 \
  --config.network.p_hsize=64 \
  --config.experiment.load_checkpoint=$CKPT \
  --config.experiment.early_stopping_patience=50 \
  --config.experiment.study=round12_sequential_v2 \
  --config.experiment.hypothesis=generalist_probe \
  --config.experiment.variant=probe_from_${RUN_LABEL} \
  --config.experiment.tags=generalist,probe,${RUN_LABEL}
```

## Decision rules

### If 12A shows meta-learning (starting cost decreases over phases)

The new architecture successfully builds transferable features across maps. This validates the sequential specialist approach and justifies HPC scale-up to 100+ maps.

### If 12B (plan_cost) transfers better than 12A (layer_delta + bonus)

Plan_cost's direct cross-layer signal outweighs its per-map bias for transfer learning. Reconsider plan_cost as the default for generalist training.

### If 12C (layer_completion) works from warm checkpoint

Sparse reward is viable when bootstrapped from a good value estimate. This is significant because sparse reward provides the least map-specific signal — potentially the best for generalization.

### If none show meta-learning despite all improvements

The sequential specialist approach is fundamentally limited. Pivot to:
1. **Fixed map pool training** (R11 recommendation: 10-20 fixed maps, train repeatedly)
2. **Surrogate-guided pre-training** (use partition function surrogate to generate training data)
3. **Behavioral cloning from parallel specialists** (train N specialists independently, distill)

## Run IDs

| Run | ID | Status |
|---|---|---|
| 12-base | pending | — |
| 12A layer_delta+bonus × 20 phases | pending | — |
| 12B plan_cost × 20 phases | pending | — |
| 12C layer_completion × 20 phases | pending | — |
| 12D remaining_cost × 20 phases | pending | — |
| Generalist probes (4×) | pending | — |

## Results

_Pending._

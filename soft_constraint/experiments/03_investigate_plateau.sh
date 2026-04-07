#!/bin/bash
# Experiment 03: Investigating the gap to direct optimizer
#
# Hypotheses:
#   H1: T=8 is reliably better than T=4 (test across 5 seeds)
#   H2: Inference-time T scaling: more infer steps = better cost (no extra training)
#   H3: Model is still improving at 3000 steps (run to 10K)
#
# Usage: bash experiments/03_investigate_plateau.sh [h1|h2|h3|all]

PYTHON=/home/marko/grid_mcts2/.venv/bin/python
COMMON="--cfg.trainer.val_check_interval=999999 --cfg.trainer.limit_val_batches=8 \
        --cfg.early_stopping.patience=999 --cfg.trainer.log_every_n_steps=1000"

RUN=${1:-all}

# ─────────────────────────────────────────────────────────────────────────────
# H1: T=4 vs T=8 across 5 seeds (2000 steps each — quick but statistically sound)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$RUN" == "h1" || "$RUN" == "all" ]]; then
  echo "=== H1: T=4 vs T=8 across seeds ==="
  for T in 4 8; do
    for SEED in 0 1 2 3 4; do
      echo "  T=$T seed=$SEED"
      WANDB_MODE=disabled $PYTHON main2.py \
        --cfg=config2.py:single_map/trm_dit/iter_T4N4 \
        --cfg.trainer.max_steps=2000 \
        --cfg.seed=$SEED \
        --cfg.model.T=$T \
        $COMMON \
        2>&1 | grep -E "val_true_cost" | tail -1
    done
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# H2: Inference-time T scaling — load best T=4 checkpoint, infer with T=4..32
# (This tests whether more refinement steps at inference is free performance)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$RUN" == "h2" || "$RUN" == "all" ]]; then
  echo "=== H2: Inference-time T scaling ==="
  # Run via inline python — loads the 3000-step T=4 checkpoint and tests T=4..32
  WANDB_MODE=disabled $PYTHON -c "
import torch, sys, json
torch.serialization.add_safe_globals([__import__('ml_collections').config_dict.config_dict.ConfigDict])
from surrogate.primitives import true_total_cost
from generalist.wrapper import NeutralAtomsWrapper, _draft_init_biased
from generalist.configs.experiments.single_map import SINGLE_MAP
from generalist.experiments.neutral_atoms.dataset import SingleMapDataset, neutral_atoms_collate_fn

ckpt = 'outputs/ca07d735/checkpoints/last.ckpt'  # T=4, 3000 steps
model = NeutralAtomsWrapper.load_from_checkpoint(ckpt, map_location='cpu')
model.eval()

ds = SingleMapDataset(SINGLE_MAP)
batch = neutral_atoms_collate_fn([next(iter(ds)) for _ in range(32)])
init_cells, task_partner, tasks_list = batch

print('infer_T, mean_cost, min_cost, results')
for T_infer in [1, 2, 4, 8, 16, 32]:
    costs = []
    for _ in range(3):  # 3 random-seed trials
        with torch.no_grad():
            plan = model.infer(init_cells, task_partner, n_steps=T_infer)
        for b in range(len(tasks_list)):
            tc, _ = true_total_cost(init_cells[b].numpy(),
                                    [plan[b,t].numpy() for t in range(3)],
                                    tasks_list[b], 5, 5)
            costs.append(tc)
    import statistics
    print(f'T={T_infer:3d}: mean={statistics.mean(costs):.2f} min={min(costs)} std={statistics.stdev(costs):.2f}')
sys.stdout.flush()
"
fi

# ─────────────────────────────────────────────────────────────────────────────
# H3: Are we still improving at 3000 steps? Run to 10K with val every 1K
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$RUN" == "h3" || "$RUN" == "all" ]]; then
  echo "=== H3: Learning curve to 10K steps ==="
  WANDB_MODE=disabled $PYTHON main2.py \
    --cfg=config2.py:single_map/trm_dit/iter_T4N4 \
    --cfg.trainer.max_steps=10000 \
    --cfg.model.T=8 \
    --cfg.trainer.val_check_interval=1000 \
    --cfg.trainer.limit_val_batches=8 \
    --cfg.early_stopping.patience=999 \
    --cfg.trainer.log_every_n_steps=500 \
    2>&1 | grep -E "val_true_cost"
fi

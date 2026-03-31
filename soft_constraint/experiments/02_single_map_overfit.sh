#!/bin/bash
# Experiment 02: Single-map overfitting test
# Can the IR model learn to optimize a fixed 5x5/12-atom/3-layer map?
#
# Map 2: atom_map=[4,12,7,11,15,5,2,21,22,23,20,8]
# Do-nothing baseline true cost: ~35
# Direct optimizer best (from soft_constraint/main.py): check outputs/
#
# Usage:
#   bash experiments/02_single_map_overfit.sh

PYTHON=/home/marko/grid_mcts2/.venv/bin/python

# Quick sanity (tiny model, 200 steps)
WANDB_MODE=disabled $PYTHON main2.py \
  --cfg=config2.py:single_map/trm_dit/tiny_T2N2 \
  --cfg.trainer.max_steps=200 \
  --cfg.trainer.val_check_interval=50 \
  --cfg.trainer.limit_val_batches=4 \
  --cfg.data.batch_size=32 \
  --cfg.optim.type=adamw \
  --cfg.optim.lr=3e-4 \
  --cfg.early_stopping.patience=100

# Longer run with iter model
# WANDB_MODE=disabled $PYTHON main2.py \
#   --cfg=config2.py:single_map/trm_dit/iter_T4N4 \
#   --cfg.trainer.max_steps=5000 \
#   --cfg.trainer.val_check_interval=500 \
#   --cfg.trainer.limit_val_batches=4 \
#   --cfg.data.batch_size=32 \
#   --cfg.optim.type=adamw \
#   --cfg.optim.lr=3e-4 \
#   --cfg.early_stopping.patience=20

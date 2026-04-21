#!/bin/bash
# Iterative Refinement sanity check experiments
# Run from grid_mcts2_prior_learning directory
set -euo pipefail
PYTHON=../grid_mcts2/.venv/bin/python
OUT=./outputs_ir

echo "============================================"
echo "IR Sanity Checks"
echo "============================================"

# ---- 1. Unit tests ----
echo ""
echo "[1/5] Running soft cost unit tests..."
$PYTHON -m iterative_refinement.test_soft_cost
echo "PASSED"

# ---- 2. Smoke test: does training converge on tiny Map 1? ----
# Map 1: 4x4, 8q, 3 layers — smallest non-trivial map
# Fixed board (no random), small model, few steps
# Expect: loss decreases, solution has non-trivial moves
echo ""
echo "[2/5] Smoke test: Map 1 fixed, 500 steps..."
$PYTHON train_ir.py \
    --config.map_num=1 \
    --config.random_board=False \
    --config.training.max_steps=500 \
    --config.training.batch_size=16 \
    --config.training.eval_every=250 \
    --config.training.save_every=500 \
    --config.training.log_every=100 \
    --config.model.T=2 \
    --config.model.num_blocks=4 \
    --config.model.hidden_size=128 \
    --config.trainer.precision=bf16-mixed \
    --config.experiment.output_dir=${OUT}/sanity_map1_fixed \
    --eval_map=1

# ---- 3. Refinement ablation: T=1 vs T=4 ----
# Does iterative refinement actually help?
# Both on Map 1 fixed, same budget
echo ""
echo "[3/5] Refinement ablation: T=1 vs T=4 on Map 1..."
for T in 1 4; do
    echo "  T=${T}..."
    $PYTHON train_ir.py \
        --config.map_num=1 \
        --config.random_board=False \
        --config.training.max_steps=1000 \
        --config.training.batch_size=16 \
        --config.training.eval_every=500 \
        --config.training.save_every=1000 \
        --config.training.log_every=200 \
        --config.model.T=${T} \
        --config.model.num_blocks=4 \
        --config.model.hidden_size=128 \
        --config.trainer.precision=bf16-mixed \
        --config.experiment.output_dir=${OUT}/sanity_T${T} \
        --eval_map=1
done

# ---- 4. Random board generalist on Map 2 shape ----
# Map 2: 5x5, 12q, 3 layers — train on random boards, eval on fixed Map 2
# This tests generalization
echo ""
echo "[4/5] Generalist: random 5x5 boards, eval on fixed Map 2..."
$PYTHON train_ir.py \
    --config.map_num=2 \
    --config.random_board=True \
    --config.training.max_steps=5000 \
    --config.training.batch_size=32 \
    --config.training.eval_every=1000 \
    --config.training.save_every=5000 \
    --config.training.log_every=200 \
    --config.model.T=4 \
    --config.model.num_blocks=4 \
    --config.model.hidden_size=128 \
    --config.trainer.precision=bf16-mixed \
    --config.experiment.output_dir=${OUT}/sanity_map2_generalist \
    --eval_map=2

# ---- 5. Reconfig weight sweep ----
# How much should we weight reconfig vs gate cost?
echo ""
echo "[5/5] Reconfig weight sweep on Map 1..."
for RW in 0.1 1.0 5.0; do
    echo "  reconfig_weight=${RW}..."
    $PYTHON train_ir.py \
        --config.map_num=1 \
        --config.random_board=False \
        --config.training.max_steps=1000 \
        --config.training.batch_size=16 \
        --config.training.eval_every=500 \
        --config.training.save_every=1000 \
        --config.training.log_every=200 \
        --config.training.reconfig_weight=${RW} \
        --config.model.T=2 \
        --config.model.num_blocks=4 \
        --config.model.hidden_size=128 \
        --config.trainer.precision=bf16-mixed \
        --config.experiment.output_dir=${OUT}/sanity_rw${RW} \
        --eval_map=1
done

# ---- Summary ----
echo ""
echo "============================================"
echo "Results summary:"
echo "============================================"
for f in ${OUT}/*/solutions/*.json; do
    dir=$(basename $(dirname $(dirname $f)))
    moves=$($PYTHON -c "import json; d=json.load(open('$f')); print(sum(len(l) for l in d['plan']))")
    cost=$($PYTHON -c "
import json, torch
from neutral_atoms.config import MAPS, _resolve_map, atom_map_to_positions
from neutral_atoms.moves import count_groups
from neutral_atoms.tasks import gates_to_moves
d = json.load(open('$f'))
H, W = d['board']['rows'], d['board']['cols']
Q = len(d['board']['initialAtoms'])
pos = {int(k): (v['row'], v['col']) for k, v in d['board']['initialAtoms'].items()}
atom_pos = torch.tensor([pos[i] for i in range(Q)], dtype=torch.long)
total = 0
for li, layer_moves in enumerate(d['plan']):
    if layer_moves:
        mt = torch.tensor([[m['from']['row'], m['from']['col'], m['to']['row'], m['to']['col']] for m in layer_moves], dtype=torch.long)
        total += count_groups(mt, canonicalize=False)
        for m in layer_moves:
            atom_pos[m['atom']] = torch.tensor([m['to']['row'], m['to']['col']])
    gm = gates_to_moves(d['circuit'][li], atom_pos)
    if len(gm) > 0:
        total += 2 * count_groups(gm, canonicalize=True)
print(total)
")
    echo "  ${dir}: cost=${cost}, moves=${moves}"
done
echo ""
echo "Visualize with atom-viz:"
echo "  ls ${OUT}/*/solutions/*.json"
echo "============================================"

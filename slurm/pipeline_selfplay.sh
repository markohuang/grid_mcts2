#!/bin/bash
# Pipeline selfplay job (CPU array). One task per shard.
# Env vars set by the submitter:
#   PIPELINE_DIR — absolute path of the pipeline's root dir
#   CYCLE        — integer index (>= 1; cycle 0 is pretrain-only)
# #SBATCH settings (account, time, array, mem) are provided by the submitter
# via sbatch CLI, so this script has no #SBATCH headers.
set -euo pipefail

: "${PIPELINE_DIR:?PIPELINE_DIR must be set}"
: "${CYCLE:?CYCLE must be set}"

# shellcheck disable=SC1091
source "$PIPELINE_DIR/pipeline_params.sh"

CYCLE_PADDED=$(printf "%02d" "$CYCLE")
PREV_PADDED=$(printf "%02d" $((CYCLE - 1)))
VENV_PY="$REPO_ROOT/.venv/bin/python"
DATASET_DIR="$PIPELINE_DIR/cycle_$CYCLE_PADDED"
WEIGHTS_PATH="$PIPELINE_DIR/checkpoints/cycle_$PREV_PADDED.ckpt"
NODE_ID="c${CYCLE_PADDED}_task${SLURM_ARRAY_TASK_ID}"

cd "$REPO_ROOT"
module load StdEnv/2023
module load python/3.11 scipy-stack

echo "=== pipeline_selfplay ==="
echo "  PIPELINE_ID=$PIPELINE_ID  CYCLE=$CYCLE  TASK=$SLURM_ARRAY_TASK_ID  NODE_ID=$NODE_ID"
echo "  host=$(hostname)  slurm_job=$SLURM_JOB_ID  array=$SLURM_ARRAY_JOB_ID"
echo "  WEIGHTS_PATH=$WEIGHTS_PATH"
echo "  DATASET_DIR=$DATASET_DIR"

# Only task 0 appends the "started" event (avoid 20x duplicate rows).
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    $VENV_PY -c "from neutral_atoms.pipeline import append_event, write_state; \
append_event('$PIPELINE_DIR', 'cycle_${CYCLE_PADDED}_selfplay_started', \
slurm_job_id='$SLURM_JOB_ID', array_job_id='$SLURM_ARRAY_JOB_ID', num_tasks='$NUM_TASKS'); \
write_state('$PIPELINE_DIR')"
fi

# shellcheck disable=SC2086
$VENV_PY selfplay_worker.py \
    --preset="$PRESET" \
    --dataset_dir="$DATASET_DIR" \
    --num_games="$GAMES_PER_TASK" \
    --num_workers="$((SELFPLAY_CPUS - 2))" \
    --batch_size="$((SELFPLAY_CPUS - 2))" \
    --weights_path="$WEIGHTS_PATH" \
    --node_id="$NODE_ID" \
    --slurm_job_id="$SLURM_JOB_ID" \
    --slurm_array_task_id="$SLURM_ARRAY_TASK_ID" \
    --notes="pipeline_${PIPELINE_ID}_cycle_${CYCLE_PADDED}" \
    $CONFIG_FLAGS

# Only task 0 appends the "completed" event. (Technically the array as a whole
# only completes when all tasks do, but afterok on the array job id handles
# the actual gating; this event is for human-readable state.)
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    $VENV_PY -c "from neutral_atoms.pipeline import append_event, write_state; \
append_event('$PIPELINE_DIR', 'cycle_${CYCLE_PADDED}_selfplay_task0_completed', \
slurm_job_id='$SLURM_JOB_ID', dataset_dir='$DATASET_DIR'); \
write_state('$PIPELINE_DIR')"
fi

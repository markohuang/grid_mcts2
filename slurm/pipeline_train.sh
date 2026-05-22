#!/bin/bash
# Pipeline training job (GPU). Called for each training cycle: the pretrain on
# bootstrap data (CYCLE=0), and every subsequent train-after-selfplay step.
#
# Env vars set by the submitter (pipeline_kickoff.py or the previous train job):
#   PIPELINE_DIR  — absolute path of the pipeline's root dir
#   CYCLE         — integer index, zero-padded when stamped on files
#
# Everything else is read from $PIPELINE_DIR/pipeline_params.sh (frozen at kickoff).
#
# On success this job submits the next (selfplay_{N+1}, train_{N+1}) pair if
# CYCLE < FINAL_CYCLE, using the A2 submit-next pattern.
#SBATCH --job-name=pipeline_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/pipeline_train_%j.out
# cpus-per-task was 4: DataLoader auto-scales to min(32, os.cpu_count()), capping
# at 4 starved the loader on 1.2M-transition cycles. 16 gives us 16 parallel
# DataLoader workers (game_to_tensordict per worker = pure CPU work, scales well).
set -euo pipefail

: "${PIPELINE_DIR:?PIPELINE_DIR must be set}"
: "${CYCLE:?CYCLE must be set}"

# shellcheck disable=SC1091
source "$PIPELINE_DIR/pipeline_params.sh"

CYCLE_PADDED=$(printf "%02d" "$CYCLE")
VENV_PY="$REPO_ROOT/.venv/bin/python"

cd "$REPO_ROOT"
module load StdEnv/2023
module load python/3.11 scipy-stack cuda cudnn

echo "=== pipeline_train ==="
echo "  PIPELINE_ID=$PIPELINE_ID  CYCLE=$CYCLE  FINAL_CYCLE=$FINAL_CYCLE"
echo "  host=$(hostname)  slurm_job=$SLURM_JOB_ID"
nvidia-smi -L || true

# --- Compose dataset_dirs (sliding window over chronological order) ---
ALL_DIRS=()
[ -n "$BOOTSTRAP_DATA" ] && ALL_DIRS+=("$BOOTSTRAP_DATA")
for i in $(seq 1 "$CYCLE"); do
    cy=$(printf "%02d" "$i")
    ALL_DIRS+=("$PIPELINE_DIR/cycle_$cy")
done
START=$(( ${#ALL_DIRS[@]} - SLIDING_WINDOW ))
[ "$START" -lt 0 ] && START=0
WINDOW_DIRS=("${ALL_DIRS[@]:$START}")
DATASET_DIRS=$(IFS=,; echo "${WINDOW_DIRS[*]}")

# --- Resolve load/save ckpts and train_epochs ---
if [ "$CYCLE" -eq 0 ]; then
    LOAD_CKPT=""  # bootstrap_ckpt could be wired through here later if needed
    TRAIN_EPOCHS_FOR_THIS_JOB=$PRETRAIN_EPOCHS
    PHASE=pretrain
else
    PREV=$(printf "%02d" $((CYCLE - 1)))
    LOAD_CKPT="$PIPELINE_DIR/checkpoints/cycle_$PREV.ckpt"
    TRAIN_EPOCHS_FOR_THIS_JOB=$TRAIN_EPOCHS
    PHASE=train
fi
SAVE_CKPT="$PIPELINE_DIR/checkpoints/cycle_$CYCLE_PADDED.ckpt"

$VENV_PY -c "from neutral_atoms.pipeline import append_event, write_state; \
append_event('$PIPELINE_DIR', 'cycle_${CYCLE_PADDED}_${PHASE}_started', \
slurm_job_id='$SLURM_JOB_ID', host='$(hostname)'); \
write_state('$PIPELINE_DIR')"

echo "  DATASET_DIRS=$DATASET_DIRS"
echo "  LOAD_CKPT=${LOAD_CKPT:-<random init>}"
echo "  SAVE_CKPT=$SAVE_CKPT"
echo "  TRAIN_EPOCHS=$TRAIN_EPOCHS_FOR_THIS_JOB"

# --- Run training ---
# shellcheck disable=SC2086
$VENV_PY train_offline.py \
    --preset="$PRESET" \
    --dataset_dirs="$DATASET_DIRS" \
    --load_ckpt="$LOAD_CKPT" \
    --save_ckpt="$SAVE_CKPT" \
    --train_epochs="$TRAIN_EPOCHS_FOR_THIS_JOB" \
    --run_id="$PIPELINE_ID" \
    --epoch="$CYCLE" \
    $CONFIG_FLAGS

$VENV_PY -c "from neutral_atoms.pipeline import append_event, write_state; \
append_event('$PIPELINE_DIR', 'cycle_${CYCLE_PADDED}_${PHASE}_completed', \
slurm_job_id='$SLURM_JOB_ID', ckpt='$SAVE_CKPT'); \
write_state('$PIPELINE_DIR')"

# --- Submit-next chain (A2 pattern) ---
if [ "$CYCLE" -lt "$FINAL_CYCLE" ]; then
    NEXT=$((CYCLE + 1))
    NEXT_PADDED=$(printf "%02d" "$NEXT")
    echo ""
    echo "=== Submitting cycle_${NEXT_PADDED} (selfplay + train) ==="

    SELFPLAY_GRES_ARG=()
    [ -n "${SELFPLAY_GRES:-}" ] && SELFPLAY_GRES_ARG=(--gres="$SELFPLAY_GRES")
    SELFPLAY_JOB_ID=$( \
        PIPELINE_DIR="$PIPELINE_DIR" CYCLE="$NEXT" \
        sbatch --parsable \
          --account="$SLURM_ACCOUNT" \
          --array="0-$((NUM_TASKS - 1))" \
          --time="$SELFPLAY_TIME" \
          --cpus-per-task="$SELFPLAY_CPUS" \
          --mem="$SELFPLAY_MEM" \
          "${SELFPLAY_GRES_ARG[@]}" \
          --job-name="pipe_sp_${PIPELINE_ID}_c${NEXT_PADDED}" \
          --output="slurm/logs/pipe_sp_${PIPELINE_ID}_c${NEXT_PADDED}_%A_%a.out" \
          "$REPO_ROOT/slurm/pipeline_selfplay.sh" \
    )
    echo "  submitted selfplay array: $SELFPLAY_JOB_ID"

    NEXT_TRAIN_JOB_ID=$( \
        PIPELINE_DIR="$PIPELINE_DIR" CYCLE="$NEXT" \
        sbatch --parsable \
          --account="$SLURM_ACCOUNT" \
          --dependency="afterok:$SELFPLAY_JOB_ID" \
          --time="$TRAIN_TIME" \
          --mem="$TRAIN_MEM" \
          --job-name="pipe_tr_${PIPELINE_ID}_c${NEXT_PADDED}" \
          --output="slurm/logs/pipe_tr_${PIPELINE_ID}_c${NEXT_PADDED}_%j.out" \
          "$REPO_ROOT/slurm/pipeline_train.sh" \
    )
    echo "  submitted next train: $NEXT_TRAIN_JOB_ID (afterok:$SELFPLAY_JOB_ID)"

    $VENV_PY -c "from neutral_atoms.pipeline import append_event, write_state; \
append_event('$PIPELINE_DIR', 'cycle_${NEXT_PADDED}_selfplay_launched', slurm_job_id='$SELFPLAY_JOB_ID'); \
append_event('$PIPELINE_DIR', 'cycle_${NEXT_PADDED}_train_launched', slurm_job_id='$NEXT_TRAIN_JOB_ID', depends_on='$SELFPLAY_JOB_ID'); \
write_state('$PIPELINE_DIR')"
else
    echo "=== Final cycle reached (CYCLE=$CYCLE = FINAL_CYCLE). Pipeline done. ==="
    $VENV_PY -c "from neutral_atoms.pipeline import append_event, write_state; \
append_event('$PIPELINE_DIR', 'pipeline_completed', pipeline_id='$PIPELINE_ID'); \
write_state('$PIPELINE_DIR')"
fi

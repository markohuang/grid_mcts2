#!/bin/bash
# Monitor Round 12 sequential training progress.
# Usage: ./monitor_r12.sh [interval_seconds]
# Default: checks every 300s (5 min)

INTERVAL=${1:-300}

echo "=== R12 Monitor (every ${INTERVAL}s) ==="
echo "Press Ctrl+C to stop."
echo ""

while true; do
    echo "────────────────────────────────────────────────────────────"
    date
    echo ""

    # Check running python processes
    RUNNING=$(ps aux | grep "main.py" | grep -v grep | grep "round12" | wc -l)
    echo "Active R12 training processes: $RUNNING"

    # GPU utilization
    echo ""
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPU"

    # Check log files for each run label
    for LABEL in 12A 12B 12C 12D base; do
        LOGFILE="experiments/r12_${LABEL}.log"
        if [ -f "$LOGFILE" ]; then
            LAST_PHASE=$(grep "^--- Phase" "$LOGFILE" | tail -1 | sed 's/.*Phase \([0-9]*\).*/\1/')
            LAST_SUMMARY=$(grep "run=" "$LOGFILE" | tail -1)
            echo ""
            echo "  ${LABEL}: phase ${LAST_PHASE:-0}/20"
            [ -n "$LAST_SUMMARY" ] && echo "  $LAST_SUMMARY"
        fi
    done

    # Check latest run's metrics for active training
    LATEST_RUN=$(ls -td outputs/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_RUN" ] && [ -f "${LATEST_RUN}metrics.jsonl" ]; then
        MANIFEST="${LATEST_RUN}manifest.json"
        if [ -f "$MANIFEST" ]; then
            STUDY=$(python3 -c "import json; print(json.load(open('$MANIFEST')).get('study','?'))" 2>/dev/null)
            VARIANT=$(python3 -c "import json; print(json.load(open('$MANIFEST')).get('variant','?'))" 2>/dev/null)
        fi
        N_EPOCHS=$(wc -l < "${LATEST_RUN}metrics.jsonl")
        LAST_LINE=$(tail -1 "${LATEST_RUN}metrics.jsonl" 2>/dev/null)
        if [ -n "$LAST_LINE" ]; then
            echo ""
            echo "  Active run: $(basename $LATEST_RUN) [${STUDY}/${VARIANT}]"
            echo "  Epochs logged: $N_EPOCHS"
            echo "$LAST_LINE" | python3 -c "
import json, sys
d = json.loads(sys.stdin.readline())
print(f'  Latest: ep{d[\"epoch\"]}, avg={d[\"avg_cost\"]:.1f}, best={d[\"best_cost\"]}, ent_l0={d.get(\"entropy_layer0\",0):.2f}, depth={d.get(\"avg_mcts_depth\",0):.1f}, grad={d.get(\"train_grad_norm\",0):.2f}')
" 2>/dev/null
        fi
    fi

    # Health checks
    echo ""
    echo "  Health checks:"

    # Check for entropy collapse (< 0.1)
    if [ -n "$LAST_LINE" ]; then
        python3 -c "
import json, sys
d = json.loads('$LAST_LINE')
ent = d.get('entropy_layer0', 1.0)
grad = d.get('train_grad_norm', 0.0)
avg = d.get('avg_cost', 99)
if ent < 0.1:
    print('  ⚠ ENTROPY COLLAPSE: ent_l0={:.3f}'.format(ent))
elif ent < 0.3:
    print('  ⚠ Low entropy: ent_l0={:.3f}'.format(ent))
else:
    print('  ✓ Entropy healthy: {:.2f}'.format(ent))
if grad > 50:
    print('  ⚠ High grad norm: {:.1f}'.format(grad))
elif grad < 0.01 and d.get('epoch', 0) > 5:
    print('  ⚠ Near-zero grads: {:.4f}'.format(grad))
else:
    print('  ✓ Grad norm: {:.2f}'.format(grad))
if avg > 20:
    print('  ⚠ High avg cost: {:.1f}'.format(avg))
else:
    print('  ✓ Avg cost: {:.1f}'.format(avg))
" 2>/dev/null
    fi

    echo ""
    sleep $INTERVAL
done

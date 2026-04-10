#!/bin/bash
# Run after round01 jobs complete to analyze results
# Usage: bash slurm/round01_analyze.sh [dataset_dir]

DATASET_DIR=${1:-~/scratch/grid_mcts2/datasets/round01_fakenet}

cd ~/grid_mcts2_prior_learning
source ~/projects/def-CHANGEME/venvs/grid_mcts2/bin/activate

python -c "
from neutral_atoms.data import dataset_stats, query_index

print('='*60)
print('DATASET OVERVIEW')
print('='*60)
dataset_stats('$DATASET_DIR')

print()
print('='*60)
print('PER REWARD MODE BREAKDOWN')
print('='*60)
df = query_index('$DATASET_DIR', '''
    SELECT
        reward_mode,
        search_bonus_weight as bonus,
        count(*) as games,
        avg(cost) as avg_cost,
        min(cost) as min_cost,
        max(cost) as max_cost,
        avg(policy_entropy) as avg_entropy,
        avg(mcts_depth) as avg_depth,
        avg(game_time_s) as avg_time_s,
        sum(steps) as total_steps
    FROM __INDEX__
    GROUP BY reward_mode, search_bonus_weight
    ORDER BY reward_mode
''')
print(df.to_string(index=False))

print()
print('='*60)
print('COST DISTRIBUTION (percentiles)')
print('='*60)
df2 = query_index('$DATASET_DIR', '''
    SELECT
        reward_mode,
        percentile_cont(0.10) WITHIN GROUP (ORDER BY cost) as p10,
        percentile_cont(0.25) WITHIN GROUP (ORDER BY cost) as p25,
        percentile_cont(0.50) WITHIN GROUP (ORDER BY cost) as p50,
        percentile_cont(0.75) WITHIN GROUP (ORDER BY cost) as p75,
        percentile_cont(0.90) WITHIN GROUP (ORDER BY cost) as p90
    FROM __INDEX__
    GROUP BY reward_mode
    ORDER BY reward_mode
''')
print(df2.to_string(index=False))

print()
print('='*60)
print('WORKER THROUGHPUT')
print('='*60)
df3 = query_index('$DATASET_DIR', '''
    SELECT
        split_part(game_id, '_', 1) || '_' || split_part(game_id, '_', 2) as worker,
        count(*) as games,
        sum(game_time_s) as total_game_time,
        avg(game_time_s) as avg_game_time
    FROM __INDEX__
    GROUP BY worker
    ORDER BY worker
''')
print(df3.to_string(index=False))
"

"""Cross-wave diagnostic script for self-play data quality.

Reads a parquet-indexed dataset and reports distribution stats, correlations,
and top-K structural gaps. Compare outputs against the thresholds in
docs/selfplay/selfplay_waves.md to judge warmstart-readiness.

Usage:
  python -m neutral_atoms.check_wave_health <dataset_dir>
  python -m neutral_atoms.check_wave_health <ds_1> <ds_2> ...  # side-by-side

Example:
  python -m neutral_atoms.check_wave_health \\
    /project/rrg-aspuru/huang651/grid_mcts2/datasets/wave01 \\
    /project/rrg-aspuru/huang651/grid_mcts2/datasets/wave02d_temp_layerdelta
"""

import math
import sys
import os
import numpy as np

from neutral_atoms.data import query_index


def _corr(df, a, b):
    return float(np.corrcoef(df[a], df[b])[0, 1])


def _wave_label(path):
    return os.path.basename(path.rstrip('/'))


def _cost_row(df):
    return {
        'min': int(df.cost.min()),
        'q10': int(df.cost.quantile(.10)),
        'q25': int(df.cost.quantile(.25)),
        'median': int(df.cost.quantile(.50)),
        'q75': int(df.cost.quantile(.75)),
        'max': int(df.cost.max()),
        'mean': float(df.cost.mean()),
    }


def _search_row(df):
    return {
        'mdepth': float(df.mcts_depth.mean()),
        'dstd': float(df.mcts_depth.std()),
        'entropy': float(df.policy_entropy.median()),
        'game_time': float(df.game_time_s.median()),
        'reward_mode': df.reward_mode.unique()[0] if df.reward_mode.nunique() == 1 else '(mixed)',
    }


def _corr_row(df):
    return {
        'root_v~cost': _corr(df, 'root_value', 'cost'),
        'entropy~cost': _corr(df, 'policy_entropy', 'cost'),
        'depth~cost': _corr(df, 'mcts_depth', 'cost'),
        'noop~cost': _corr(df, 'noop_frac', 'cost'),
    }


def _topk_gap_row(df, k=100):
    top = df.nsmallest(k, 'cost')
    return {
        'top_cost': float(top.cost.median()),
        'top_entropy': float(top.policy_entropy.median()),
        'overall_entropy': float(df.policy_entropy.median()),
        'entropy_gap': float(df.policy_entropy.median() - top.policy_entropy.median()),
        'top_depth': float(top.mcts_depth.mean()),
        'overall_depth': float(df.mcts_depth.mean()),
        'depth_gap': float(top.mcts_depth.mean() - df.mcts_depth.mean()),
    }


def _diversity_row(df, k=100):
    # Proxy for trajectory uniqueness: distinct (cost, move_distance, noop_frac) triples.
    # Different action sequences almost always differ in at least one of these.
    # Not a true trajectory hash (we don't store history in the index), but a strong lower
    # bound on diversity -- colliding triples can hide same-trajectory duplicates, not split them.
    triples = list(zip(df.cost, df.move_distance, df.noop_frac.round(3)))
    unique_frac = len(set(triples)) / len(df) if len(df) > 0 else 0.0
    top = df.nsmallest(k, 'cost')
    top_triples = list(zip(top.cost, top.move_distance, top.noop_frac.round(3)))
    top_unique_frac = len(set(top_triples)) / max(len(top), 1)
    # relative cost spread: (q90 - q10) / median. Scale-invariant, handles tightening at lower
    # absolute costs (where q90-q10 drops mechanically as the distribution floor approaches optimum).
    median = float(df.cost.quantile(.50))
    rel_spread = float((df.cost.quantile(.9) - df.cost.quantile(.1)) / max(median, 1))
    return {
        'unique_frac': float(unique_frac),
        'top_unique_frac': float(top_unique_frac),
        'rel_spread': rel_spread,
        'n_trajs_at_min': int(len(set(zip(df[df.cost == df.cost.min()].move_distance,
                                          df[df.cost == df.cost.min()].noop_frac.round(3))))),
    }


def _inner_row(df):
    # reached_terminal_frac may be NaN for waves generated before the metric was added (2026-04-20+).
    reached = float(df.mcts_reached_terminal_frac.mean()) if 'mcts_reached_terminal_frac' in df else float('nan')
    return {
        'rew_std': float(df.mcts_reward_std.mean()),
        'rew_mean': float(df.mcts_reward_sum_mean.mean()),
        'rew_abs': float(df.mcts_reward_abs.mean()),
        'boundfr': float(df.mcts_boundary_frac.mean()),
        'termfr': reached,
        'signch': float(df.mcts_sign_changes.mean()),
        'rewfrac': float(df.mcts_reward_frac.mean()),
    }


# Thresholds for a wave to be "warmstart-viable". See docs/selfplay/selfplay_waves.md
# for rationale. Convention: (direction, min_viable, stretch)
#   direction='low'  -> lower value is better (threshold is a ceiling)
#   direction='high' -> higher value is better (threshold is a floor)
#
# NOTE: depth-related thresholds (mdepth_mean, mdepth_std) are currently calibrated
# for map 2 (5x5_12qb_4gpl_3lyrs, 24-step episodes). They'll need re-scaling for
# larger boards — see docs/selfplay/selfplay_waves.md §"Thresholds across map scales".
THRESHOLDS = {
    'cost_median':             ('low',  25,    18),
    'cost_min':                ('low',  18,    13),
    'rel_cost_spread':         ('high', 0.30,  0.60),  # (q90-q10)/median; replaces absolute cost_spread
    'policy_entropy':          ('low',  1.5,   1.0),   # at tau=1; not directly comparable across tau
    'mdepth_mean':             ('high', 4.0,   5.5),
    'mdepth_std':              ('high', 0.30,  0.80),  # problem-adaptive depth; map-scale dependent
    'corr_depth_cost':         ('low',  -0.20, -0.40),
    'corr_entropy_cost':       ('high', 0.20,  0.40),
    'corr_root_v_cost':        ('low',  0.00,  -0.50),
    'corr_noop_cost':          ('low',  -0.20, -0.40),
    'reached_terminal_frac':   ('high', 0.50,  0.90),  # added 2026-04-20; see mcts.py
    'topk_entropy_gap':        ('high', 0.30,  0.70),  # overall - top
    'topk_depth_gap':          ('high', 0.10,  0.50),  # top - overall
    # Training-data-quality (diversity) rows: added 2026-04-20. Proxies since we don't index
    # trajectory hashes. unique_triples_frac > 0.20 means most of the 20k games contribute
    # distinct training targets; much lower means heavy replication.
    'unique_triples_frac':     ('high', 0.20,  0.40),  # wave-level diversity
    'top100_unique_frac':      ('high', 0.80,  0.95),  # good-games diversity
}


def _status(direction, value, min_viable, stretch):
    if isinstance(value, float) and math.isnan(value):
        return 'N/A'
    if direction == 'low':
        if value <= stretch: return 'STRETCH'
        if value <= min_viable: return 'MIN'
        return 'FAIL'
    else:
        if value >= stretch: return 'STRETCH'
        if value >= min_viable: return 'MIN'
        return 'FAIL'


def _threshold_checks(df):
    cost = _cost_row(df)
    search = _search_row(df)
    corrs = _corr_row(df)
    topk = _topk_gap_row(df)
    inner = _inner_row(df)

    div = _diversity_row(df)
    checks = [
        ('cost_median',           cost['median']),
        ('cost_min',              cost['min']),
        ('rel_cost_spread',       div['rel_spread']),
        ('policy_entropy',        search['entropy']),
        ('mdepth_mean',           search['mdepth']),
        ('mdepth_std',            search['dstd']),
        ('corr_depth_cost',       corrs['depth~cost']),
        ('corr_entropy_cost',     corrs['entropy~cost']),
        ('corr_root_v_cost',      corrs['root_v~cost']),
        ('corr_noop_cost',        corrs['noop~cost']),
        ('reached_terminal_frac', inner['termfr']),
        ('topk_entropy_gap',      topk['entropy_gap']),
        ('topk_depth_gap',        topk['depth_gap']),
        ('unique_triples_frac',   div['unique_frac']),
        ('top100_unique_frac',    div['top_unique_frac']),
    ]
    return checks


def print_table(title, rows, headers):
    print(f'\n=== {title} ===')
    # col widths
    widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = ' '.join('{:>' + str(w) + '}' for w in widths)
    print(fmt.format(*headers))
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))


def fmt_f(x, n=3):
    return f'{x:+.{n}f}' if isinstance(x, float) else str(x)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    paths = sys.argv[1:]
    dfs = []
    for p in paths:
        df = query_index(p)
        dfs.append((_wave_label(p), df))
        print(f'{_wave_label(p):<30} {len(df)} rows')

    # ----- distributions -----
    print_table('COST', [
        [name,
         _cost_row(df)['min'],
         _cost_row(df)['q10'],
         _cost_row(df)['q25'],
         _cost_row(df)['median'],
         _cost_row(df)['q75'],
         _cost_row(df)['max'],
         f"{_cost_row(df)['mean']:.2f}"]
        for name, df in dfs
    ], ['wave', 'min', 'q10', 'q25', 'median', 'q75', 'max', 'mean'])

    print_table('SEARCH', [
        [name,
         f"{_search_row(df)['mdepth']:.3f}",
         f"{_search_row(df)['dstd']:.3f}",
         f"{_search_row(df)['entropy']:.3f}",
         f"{_search_row(df)['game_time']:.1f}",
         _search_row(df)['reward_mode']]
        for name, df in dfs
    ], ['wave', 'mdepth', 'dstd', 'entropy', 'game_time', 'reward_mode'])

    print_table('CORRELATIONS', [
        [name,
         fmt_f(_corr_row(df)['root_v~cost'], 4),
         fmt_f(_corr_row(df)['entropy~cost'], 4),
         fmt_f(_corr_row(df)['depth~cost'], 4),
         fmt_f(_corr_row(df)['noop~cost'], 4)]
        for name, df in dfs
    ], ['wave', 'root_v~cost', 'entropy~cost', 'depth~cost', 'noop~cost'])

    print_table('TOP-100 vs OVERALL', [
        [name,
         int(_topk_gap_row(df)['top_cost']),
         f"{_topk_gap_row(df)['top_entropy']:.3f}",
         f"{_topk_gap_row(df)['overall_entropy']:.3f}",
         f"{_topk_gap_row(df)['entropy_gap']:+.3f}",
         f"{_topk_gap_row(df)['top_depth']:.3f}",
         f"{_topk_gap_row(df)['overall_depth']:.3f}",
         f"{_topk_gap_row(df)['depth_gap']:+.3f}"]
        for name, df in dfs
    ], ['wave', 'top_cost', 'top_ent', 'all_ent', 'ent_gap', 'top_dep', 'all_dep', 'dep_gap'])

    print_table('DIVERSITY (training-data quality proxy)', [
        [name,
         f"{_diversity_row(df)['unique_frac']:.3f}",
         f"{_diversity_row(df)['top_unique_frac']:.3f}",
         f"{_diversity_row(df)['rel_spread']:.3f}",
         _diversity_row(df)['n_trajs_at_min']]
        for name, df in dfs
    ], ['wave', 'uniqfr', 'top_uniqfr', 'rel_spread', 'trajs_at_min'])

    print_table('INNER', [
        [name,
         f"{_inner_row(df)['rew_std']:.3f}",
         f"{_inner_row(df)['rew_mean']:.3f}",
         f"{_inner_row(df)['rew_abs']:.3f}",
         f"{_inner_row(df)['boundfr']:.3f}",
         (f"{_inner_row(df)['termfr']:.3f}" if not math.isnan(_inner_row(df)['termfr']) else 'N/A'),
         f"{_inner_row(df)['signch']:.3f}",
         f"{_inner_row(df)['rewfrac']:.3f}"]
        for name, df in dfs
    ], ['wave', 'rew_std', 'rew_mean', 'rew_abs', 'boundfr', 'termfr', 'signch', 'rewfrac'])

    # ----- threshold gate -----
    print('\n=== THRESHOLD GATE (see docs/selfplay/selfplay_waves.md for rationale) ===')
    print(f"{'metric':<22} ", end='')
    for name, _ in dfs:
        print(f'{name[:15]:>17}', end='')
    print()
    for name, df in dfs:
        pass  # just the header
    for metric, _ in _threshold_checks(dfs[0][1]):
        direction, min_v, stretch = THRESHOLDS[metric]
        line = f'{metric:<22} '
        for name, df in dfs:
            checks = dict(_threshold_checks(df))
            val = checks[metric]
            st = _status(direction, val, min_v, stretch)
            mark = {'STRETCH': '++', 'MIN': ' +', 'FAIL': '  ', 'N/A': 'na'}[st]
            if isinstance(val, float) and math.isnan(val):
                line += f'{"N/A":>12} {mark:<4}'
            elif isinstance(val, float):
                line += f'{val:>+12.3f} {mark:<4}'
            else:
                line += f'{val:>12} {mark:<4}'
        print(line)

    print()
    print('Legend: ++ = stretch target; + = min viable; blank = fail')
    print('Thresholds defined in neutral_atoms/check_wave_health.py THRESHOLDS dict.')
    print('Expand or adjust with new evidence from waves.')


if __name__ == '__main__':
    main()

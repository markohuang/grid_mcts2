"""Phase 2 A/B report: control (pUCT) vs treatment (Gumbel) side-by-side.

Reads the parquet index of each wave, prints the cost distribution, search-quality
signals, correlation table, and wallclock comparison. Emits a pass/fail verdict
per the decision rule in docs/gumbel_pczero_plan.md §3.

Usage:
  python scripts/gumbel_ab_report.py <control_dir> <treatment_dir>
  python scripts/gumbel_ab_report.py \\
    /scratch/huang651/grid_mcts2/datasets/gumbel_a_control \\
    /scratch/huang651/grid_mcts2/datasets/gumbel_a_treatment
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neutral_atoms.data import query_index


def _corr(df, a, b):
    if df[a].nunique() < 2 or df[b].nunique() < 2:
        return float('nan')
    return float(np.corrcoef(df[a], df[b])[0, 1])


def _row(df):
    return {
        'n': len(df),
        'cost_min': int(df.cost.min()),
        'cost_q10': int(df.cost.quantile(.10)),
        'cost_median': int(df.cost.quantile(.50)),
        'cost_q75': int(df.cost.quantile(.75)),
        'cost_max': int(df.cost.max()),
        'cost_mean': float(df.cost.mean()),
        'mcts_depth_mean': float(df.mcts_depth.mean()),
        'mcts_depth_std': float(df.mcts_depth.std()),
        'policy_entropy_median': float(df.policy_entropy.median()),
        'reached_terminal_frac': float(df.mcts_reached_terminal_frac.mean()),
        'game_time_s_median': float(df.game_time_s.median()),
        'unique_maps': df.map_id.nunique(),
        'corr_depth_cost': _corr(df, 'mcts_depth', 'cost'),
        'corr_entropy_cost': _corr(df, 'policy_entropy', 'cost'),
        'corr_rootv_cost': _corr(df, 'root_value', 'cost'),
    }


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <control_dir> <treatment_dir>", file=sys.stderr)
        sys.exit(1)
    ctrl_dir, trmt_dir = sys.argv[1], sys.argv[2]

    ctrl = query_index(ctrl_dir)
    trmt = query_index(trmt_dir)
    ctrl_row = _row(ctrl)
    trmt_row = _row(trmt)

    print(f"\n=== Gumbel Phase 2 A/B ===")
    print(f"  control   : {ctrl_dir}  (n={ctrl_row['n']})")
    print(f"  treatment : {trmt_dir}  (n={trmt_row['n']})\n")

    keys = list(ctrl_row.keys())
    width = max(len(k) for k in keys)
    print(f"  {'metric':<{width}}  {'control':>14}  {'treatment':>14}  delta")
    print(f"  {'-'*width}  {'-'*14}  {'-'*14}  -----")
    for k in keys:
        c, t = ctrl_row[k], trmt_row[k]
        if isinstance(c, (int, float)) and isinstance(t, (int, float)):
            d = t - c
            ds = f"{d:+.3f}" if isinstance(d, float) else f"{d:+d}"
        else:
            ds = "-"
        print(f"  {k:<{width}}  {_fmt(c):>14}  {_fmt(t):>14}  {ds}")

    # Decision rule (docs/gumbel_pczero_plan.md §3)
    print("\n=== Decision rule (plan §3) ===")
    checks = []
    checks.append(("cost median: treatment <= control",
                   trmt_row['cost_median'] <= ctrl_row['cost_median']))
    checks.append(("corr(depth,cost) at least as negative (tol 0.05)",
                   trmt_row['corr_depth_cost'] <= ctrl_row['corr_depth_cost'] + 0.05))
    checks.append(("reached_terminal_frac within ±5pp of control",
                   abs(trmt_row['reached_terminal_frac'] - ctrl_row['reached_terminal_frac']) <= 0.05))
    checks.append(("wallclock within 1.5× control",
                   trmt_row['game_time_s_median'] <= 1.5 * ctrl_row['game_time_s_median']))
    checks.append(("map uniqueness = 100%",
                   ctrl_row['unique_maps'] == ctrl_row['n']
                   and trmt_row['unique_maps'] == trmt_row['n']))

    all_pass = True
    for desc, ok in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {desc}")
        all_pass = all_pass and ok
    print(f"\n=== Verdict: {'PASS' if all_pass else 'FAIL'} ===\n")


if __name__ == '__main__':
    main()

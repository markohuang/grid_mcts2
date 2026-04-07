"""Analyze experiment runs.

Examples:
  python analyze.py
  python analyze.py --study round08_reward_signal
  python analyze.py --hypothesis H2 --tag control
  python analyze.py d90b67b3
"""
import argparse
import json
import os


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def truncate(text, width):
    text = str(text or '')
    return text if len(text) <= width else text[:width - 1] + '…'


def fmt(value, digits=3):
    if value in (None, ''):
        return ''
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip('0').rstrip('.')
    return str(value)


def matches_filters(entry, args):
    if not args.all and entry.get('use_fake'):
        return False
    if args.study and entry.get('study') != args.study:
        return False
    if args.hypothesis and entry.get('hypothesis') != args.hypothesis:
        return False
    if args.tag and args.tag not in entry.get('tags', []):
        return False
    return True


def show_registry(args):
    entries = load_jsonl(os.path.join(args.output_dir, 'run_registry.jsonl'))
    entries = [e for e in entries if matches_filters(e, args)]
    if not entries:
        print("No matching runs in registry.")
        return
    entries.sort(key=lambda e: e.get('timestamp', ''), reverse=True)
    if args.limit:
        entries = entries[:args.limit]

    header = (
        f"{'run_id':<10} {'study':<18} {'hyp':<5} {'variant':<10} {'map':>3} "
        f"{'reward':<16} {'sims':>4} {'alpha':>5} {'rand':>4} {'aug':>3} "
        f"{'curr':>4} {'best':>5} {'avg':>5} {'ep':>4} {'tags':<16}"
    )
    print(header)
    print("-" * len(header))
    for e in entries:
        curr = e.get('curriculum_maps', 0)
        if e.get('curriculum_initial_phase', 0):
            curr = f"{curr}+{e.get('curriculum_initial_phase', 0)}"
        print(
            f"{e.get('run_id', '?'):<10} "
            f"{truncate(e.get('study', ''), 18):<18} "
            f"{truncate(e.get('hypothesis', ''), 5):<5} "
            f"{truncate(e.get('variant', ''), 10):<10} "
            f"{e.get('map_num', '?'):>3} "
            f"{truncate(e.get('reward_mode', ''), 16):<16} "
            f"{e.get('num_simulations', '?'):>4} "
            f"{e.get('root_dirichlet_alpha', '?'):>5} "
            f"{('yes' if e.get('random_board') else 'no'):>4} "
            f"{('yes' if e.get('data_augmentation') else 'no'):>3} "
            f"{str(curr):>4} "
            f"{str(e.get('best_cost', '?')):>5} "
            f"{str(e.get('avg_cost', '?')):>5} "
            f"{str(e.get('epochs_completed', '?')):>4} "
            f"{truncate(','.join(e.get('tags', [])), 16):<16}"
        )


def show_run(run_id, output_dir):
    run_dir = os.path.join(output_dir, run_id)
    metrics = load_jsonl(os.path.join(run_dir, 'metrics.jsonl'))
    manifest = load_json(os.path.join(run_dir, 'manifest.json'))
    config = load_json(os.path.join(run_dir, 'config.json'))
    if not metrics and not manifest and not config:
        print(f"No data found for run {run_id}")
        return

    study = manifest.get('study') or config.get('experiment', {}).get('study', '')
    hypothesis = manifest.get('hypothesis') or config.get('experiment', {}).get('hypothesis', '')
    variant = manifest.get('variant') or config.get('experiment', {}).get('variant', '')
    tags = manifest.get('tags') or []
    print(f"\n=== Run {run_id} ===")
    print(
        f"study={study or '-'}  hypothesis={hypothesis or '-'}  variant={variant or '-'}  "
        f"tags={','.join(tags) or '-'}"
    )
    print(
        f"map={manifest.get('map_num', config.get('map_num', '?'))}  "
        f"reward={manifest.get('reward_mode', config.get('env', {}).get('reward_mode', '?'))}  "
        f"sims={manifest.get('num_simulations', config.get('mcts', {}).get('num_simulations', '?'))}  "
        f"random_board={manifest.get('random_board', config.get('random_board', False))}"
    )
    if manifest.get('parent_run') or manifest.get('decision') or manifest.get('notes'):
        print(
            f"parent={manifest.get('parent_run') or '-'}  "
            f"decision={manifest.get('decision') or '-'}"
        )
        if manifest.get('notes'):
            print(f"notes={manifest['notes']}")
    if manifest.get('git_commit'):
        dirty = manifest.get('git_dirty')
        dirty_str = 'dirty' if dirty else 'clean'
        print(f"git={manifest['git_commit'][:12]} ({dirty_str})")
    if manifest.get('argv'):
        print(f"argv={' '.join(manifest['argv'])}")

    if not metrics:
        print("No per-epoch metrics.")
        return

    header = (
        f"{'ep':>3} {'best':>5} {'avg':>5} {'eval0':>5} {'eval0avg':>7} "
        f"{'depth':>5} {'rfrac':>5} {'bfrac':>5} {'rstd':>5} "
        f"{'entropy':>7} {'v_corr':>7} {'loss':>7}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics:
        print(
            f"{fmt(m.get('epoch', '?'), 0):>3} "
            f"{fmt(m.get('best_cost', ''), 0):>5} "
            f"{fmt(m.get('avg_cost', ''), 1):>5} "
            f"{fmt(m.get('eval_map0_best', ''), 0):>5} "
            f"{fmt(m.get('eval_map0_avg', ''), 1):>7} "
            f"{fmt(m.get('avg_mcts_depth', ''), 1):>5} "
            f"{fmt(m.get('mcts_reward_frac', ''), 2):>5} "
            f"{fmt(m.get('mcts_boundary_reach_frac', ''), 2):>5} "
            f"{fmt(m.get('mcts_reward_sum_std', ''), 2):>5} "
            f"{fmt(m.get('avg_policy_entropy', ''), 3):>7} "
            f"{fmt(m.get('val_cost_corr', ''), 3):>7} "
            f"{fmt(m.get('train_total', ''), 3):>7}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_ids', nargs='*', help='Run ids to inspect in detail')
    parser.add_argument('--output-dir', default='./outputs')
    parser.add_argument('--study')
    parser.add_argument('--hypothesis')
    parser.add_argument('--tag')
    parser.add_argument('--all', action='store_true', help='Include fake-net runs')
    parser.add_argument('--limit', type=int, default=30)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.run_ids:
        for run_id in args.run_ids:
            show_run(run_id, args.output_dir)
    else:
        show_registry(args)

"""
Experiment 03: Investigating the plateau at val_true_cost=15

H1: T=8 reliably beats T=4 (5 seeds, 2000 steps each)
H2: Inference-time scaling — more T steps at infer from a trained checkpoint
H3: Learning curve to 10K steps — still improving or plateaued?

Usage:
    python experiments/run_03.py h1
    python experiments/run_03.py h2 <run_id>   # e.g., ca07d735
    python experiments/run_03.py h3
    python experiments/run_03.py all
"""
import sys, os, json, statistics, subprocess, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
PYTHON = '/home/marko/grid_mcts2/.venv/bin/python'


def run_training(seed, max_steps, T, extra_flags=''):
    cmd = (f'WANDB_MODE=disabled {PYTHON} main2.py '
           f'--cfg=config2.py:single_map/trm_dit/iter_T4N4 '
           f'--cfg.trainer.max_steps={max_steps} '
           f'--cfg.trainer.val_check_interval={max_steps} '
           f'--cfg.trainer.limit_val_batches=8 '
           f'--cfg.early_stopping.patience=9999 '
           f'--cfg.trainer.log_every_n_steps=9999 '
           f'--cfg.seed={seed} '
           f'--cfg.model.T={T} '
           f'{extra_flags}')
    subprocess.run(cmd, shell=True, capture_output=True)
    # Read latest registry entry
    with open('outputs/run_registry.jsonl') as f:
        last = json.loads(f.readlines()[-1])
    return last['metrics'].get('val_true_cost', None), last['run_id']


def h1_T_vs_seeds():
    print('\n=== H1: T=4 vs T=8 across 5 seeds (2000 steps each) ===')
    print(f'{"T":>4}  {"seed":>4}  {"true_cost":>10}')
    print('-' * 25)
    results = {4: [], 8: []}
    for T in [4, 8]:
        for seed in range(5):
            cost, rid = run_training(seed=seed, max_steps=2000, T=T)
            results[T].append(cost)
            print(f'  T={T}  seed={seed}  cost={cost}  ({rid})')
        m = statistics.mean(results[T])
        s = statistics.stdev(results[T]) if len(results[T]) > 1 else 0
        print(f'  T={T}  SUMMARY: mean={m:.1f} ± {s:.1f}  min={min(results[T])}  max={max(results[T])}')
        print()


def h2_inference_scaling(run_id):
    print(f'\n=== H2: Inference-time T scaling (checkpoint: {run_id}) ===')
    torch.serialization.add_safe_globals(
        [__import__('ml_collections').config_dict.config_dict.ConfigDict])
    from surrogate.primitives import true_total_cost
    from generalist.wrapper import NeutralAtomsWrapper
    from generalist.configs.experiments.single_map import SINGLE_MAP
    from generalist.experiments.neutral_atoms.dataset import SingleMapDataset, neutral_atoms_collate_fn

    ckpt = f'outputs/{run_id}/checkpoints/last.ckpt'
    model = NeutralAtomsWrapper.load_from_checkpoint(ckpt, map_location='cpu')
    model.eval()

    ds = SingleMapDataset(SINGLE_MAP)
    # Use 3 different random batches of 32 to get stable estimates
    all_batches = [neutral_atoms_collate_fn([next(iter(ds)) for _ in range(32)])
                   for _ in range(3)]

    print(f'  {"T_infer":>7}  {"mean":>6}  {"std":>5}  {"min":>4}  {"max":>4}')
    print('  ' + '-' * 32)
    for T_infer in [1, 2, 4, 8, 16, 32]:
        costs = []
        for init_cells, task_partner, tasks_list in all_batches:
            with torch.no_grad():
                plan = model.infer(init_cells, task_partner, n_steps=T_infer)
            for b in range(len(tasks_list)):
                tc, _ = true_total_cost(init_cells[b].numpy(),
                                        [plan[b, t].numpy() for t in range(3)],
                                        tasks_list[b], 5, 5)
                costs.append(tc)
        m = statistics.mean(costs)
        s = statistics.stdev(costs)
        print(f'  T={T_infer:4d}:  mean={m:.2f}  std={s:.2f}  '
              f'min={min(costs)}  max={max(costs)}')


def h3_learning_curve():
    print('\n=== H3: Learning curve to 10K steps (T=8, 3 seeds) ===')
    # Run with val every 1K, parse CSV metrics
    import csv
    for seed in range(3):
        cmd = (f'WANDB_MODE=disabled {PYTHON} main2.py '
               f'--cfg=config2.py:single_map/trm_dit/iter_T4N4 '
               f'--cfg.trainer.max_steps=10000 '
               f'--cfg.model.T=8 '
               f'--cfg.trainer.val_check_interval=1000 '
               f'--cfg.trainer.limit_val_batches=8 '
               f'--cfg.early_stopping.patience=9999 '
               f'--cfg.trainer.log_every_n_steps=9999 '
               f'--cfg.seed={seed}')
        result = subprocess.run(cmd, shell=True, capture_output=True)
        # Get run_id from registry
        with open('outputs/run_registry.jsonl') as f:
            last = json.loads(f.readlines()[-1])
        run_id = last['run_id']
        # Read metrics CSV
        metrics_path = Path(f'outputs/{run_id}/metrics.csv')
        if metrics_path.exists():
            print(f'  seed={seed} ({run_id}):')
            with open(metrics_path) as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                if row.get('val_true_cost'):
                    print(f'    step={int(float(row["step"]))}'
                          f'  val_true_cost={float(row["val_true_cost"]):.0f}'
                          f'  val_loss={float(row["val_loss"]):.2f}')


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if arg == 'h1' or arg == 'all':
        h1_T_vs_seeds()

    if arg == 'h2':
        run_id = sys.argv[2] if len(sys.argv) > 2 else 'ca07d735'
        h2_inference_scaling(run_id)
    elif arg == 'all':
        # Use the best existing 3000-step T=4 run
        h2_inference_scaling('ca07d735')

    if arg == 'h3' or arg == 'all':
        h3_learning_curve()

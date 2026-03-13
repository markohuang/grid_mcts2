"""Quick analysis of experiment runs. Usage: python analyze.py [run_id ...]

No args: show registry summary. With run_ids: show per-epoch metrics for those runs.
"""
import json, sys, os

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]

def show_registry(output_dir='./outputs'):
    entries = load_jsonl(os.path.join(output_dir, 'run_registry.jsonl'))
    if not entries:
        print("No runs in registry.")
        return
    print(f"{'run_id':<10} {'map':>3} {'reward':<14} {'sims':>4} {'sp':>3} {'par':>3} "
          f"{'bs':>4} {'steps':>5} {'cw':>4} {'lw':>4} "
          f"{'best':>5} {'avg':>5} {'comp':>5} {'ep':>3} {'loss':>7}")
    print("-" * 110)
    for e in entries:
        if e.get('use_fake'):
            continue
        print(f"{e.get('run_id','?'):<10} "
              f"{e.get('map_num','?'):>3} "
              f"{e.get('reward_mode','cost_delta'):<14} "
              f"{e.get('num_simulations','?'):>4} "
              f"{e.get('num_selfplay','?'):>3} "
              f"{e.get('num_parallel_games',1):>3} "
              f"{e.get('batch_size','?'):>4} "
              f"{e.get('training_steps','?'):>5} "
              f"{e.get('correctness_weight',1.0):>4.1f} "
              f"{e.get('latency_weight',1.0):>4.1f} "
              f"{str(e.get('best_cost','?')):>5} "
              f"{str(e.get('avg_cost','?')):>5} "
              f"{e.get('completion_rate',0):>5.0%} "
              f"{e.get('epochs_completed','?'):>3} "
              f"{e.get('final_loss', 0):>7.3f}")

def show_run(run_id, output_dir='./outputs'):
    run_dir = os.path.join(output_dir, run_id)
    metrics = load_jsonl(os.path.join(run_dir, 'metrics.jsonl'))
    if not metrics:
        print(f"No metrics for {run_id}")
        return
    config = {}
    config_path = os.path.join(run_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    print(f"\n=== Run {run_id} === map={config.get('map_num')}, "
          f"reward={config.get('env',{}).get('reward_mode','?')}, "
          f"sims={config.get('mcts',{}).get('num_simulations','?')}")
    print(f"{'ep':>3} {'comp':>5} {'steps':>5} {'best':>5} {'avg':>5} "
          f"{'best_sf':>6} {'root_v':>7} {'pi_ent':>6} {'gate%':>5} "
          f"{'sp_t':>5} {'tr_t':>5} {'loss':>7}")
    print("-" * 85)
    for m in metrics:
        gate_frac = m.get('gate_action_fraction', None)
        print(f"{m['epoch']:>3} "
              f"{m.get('completion_rate',0):>5.0%} "
              f"{m.get('avg_steps',0):>5.1f} "
              f"{str(m.get('best_cost','')):>5} "
              f"{str(m.get('avg_cost','')):>5} "
              f"{str(m.get('best_cost_so_far','')):>6} "
              f"{m.get('avg_root_value',0):>7.2f} "
              f"{m.get('avg_policy_entropy',0):>6.2f} "
              f"{(f'{gate_frac:.0%}' if gate_frac is not None else ''):>5} "
              f"{m.get('selfplay_time',0):>5.1f} "
              f"{m.get('train_time',0):>5.1f} "
              f"{m.get('train_total',0):>7.3f}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for rid in sys.argv[1:]:
            show_run(rid)
    else:
        show_registry()

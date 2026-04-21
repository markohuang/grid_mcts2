import json
import os
import time
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from .config import map_class, map_id

# ---- Parquet Index Schema ----

INDEX_SCHEMA = pa.schema([
    ('game_id', pa.string()),
    ('map_class', pa.string()),
    ('map_id', pa.string()),
    ('cost', pa.int32()),
    ('steps', pa.int32()),
    ('num_moves', pa.int32()),
    ('noop_frac', pa.float32()),
    ('move_distance', pa.int32()),
    ('policy_entropy', pa.float32()),
    ('root_value', pa.float32()),
    ('mcts_depth', pa.float32()),
    ('mcts_reward_std', pa.float32()),
    ('mcts_reward_abs', pa.float32()),
    ('mcts_reward_sum_mean', pa.float32()),
    ('mcts_boundary_frac', pa.float32()),
    ('mcts_reached_terminal_frac', pa.float32()),
    ('mcts_sign_changes', pa.float32()),
    ('mcts_reward_frac', pa.float32()),
    ('game_time_s', pa.float32()),
    ('num_simulations', pa.int32()),
    ('weight_gen', pa.int64()),
    ('weight_file', pa.string()),
    ('weight_run_id', pa.string()),
    ('weight_git_sha', pa.string()),
    ('selfplay_git_sha', pa.string()),
    ('slurm_job_id', pa.string()),
    ('slurm_array_task_id', pa.string()),
    ('reward_mode', pa.string()),
    ('prior_mix_weight', pa.float32()),
    ('batch_file', pa.string()),
    ('game_idx', pa.int32()),
    ('timestamp', pa.string()),
])


def atomic_save(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f'.tmp.{os.getpid()}'
    torch.save(data, tmp)
    os.rename(tmp, path)


def save_map_spec(dataset_dir, map_data):
    mid = map_id(map_data)
    maps_dir = os.path.join(dataset_dir, 'maps')
    os.makedirs(maps_dir, exist_ok=True)
    path = os.path.join(maps_dir, f'{mid}.json')
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({
                'map_id': mid,
                'map_class': map_class(map_data),
                'board_dim': list(map_data['board_dim']),
                'num_qubits': map_data['num_qubits'],
                'atom_map': map_data['atom_map'],
                'tasks': map_data['tasks'],
            }, f, indent=2)
    return mid


def save_game_batch(dataset_dir, games_dicts, map_data, node_id, batch_seq,
                    map_id_override=None):
    mc = map_class(map_data)
    mid = map_id_override if map_id_override is not None else map_id(map_data)
    batch_dir = os.path.join(dataset_dir, 'games', mc, mid)
    os.makedirs(batch_dir, exist_ok=True)
    filename = f'batch_{node_id}_{batch_seq:06d}.pt'
    path = os.path.join(batch_dir, filename)
    atomic_save(games_dicts, path)
    return path


def save_manifest(dataset_dir, config_dict, notes=''):
    import time
    path = os.path.join(dataset_dir, 'manifest.json')
    manifest = {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'notes': notes,
        'config': config_dict,
    }
    os.makedirs(dataset_dir, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)


def scan_dataset(dataset_dir, filter_class=None, filter_map_id=None):
    games_dir = os.path.join(dataset_dir, 'games')
    if not os.path.exists(games_dir):
        return []
    batch_files = []
    for mc in os.listdir(games_dir):
        if filter_class and mc != filter_class:
            continue
        mc_dir = os.path.join(games_dir, mc)
        if not os.path.isdir(mc_dir):
            continue
        for mid in os.listdir(mc_dir):
            if filter_map_id and mid != filter_map_id:
                continue
            mid_dir = os.path.join(mc_dir, mid)
            if not os.path.isdir(mid_dir):
                continue
            for f in os.listdir(mid_dir):
                if f.endswith('.pt') and not f.endswith('.tmp'):
                    batch_files.append(os.path.join(mid_dir, f))
    return sorted(batch_files)


def load_game_batches(batch_files):
    games = []
    for path in batch_files:
        batch = torch.load(path, weights_only=False)
        games.extend(batch)
    return games


def _index_rows_from_batch(game_dicts, batch_file_rel, node_id, batch_seq):
    rows = []
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    for idx, g in enumerate(game_dicts):
        m = g.get('metrics', {})
        wl = g.get('weight_lineage', {})
        sc = g.get('selfplay_context', {})
        rows.append({
            'game_id': f'{node_id}_{batch_seq:06d}_{idx:04d}',
            'map_class': g.get('map_class', ''),
            'map_id': g.get('map_id', ''),
            'cost': g.get('cost', -1),
            'steps': len(g.get('history', [])),
            'num_moves': m.get('num_moves', 0),
            'noop_frac': m.get('noop_frac', 0.0),
            'move_distance': m.get('move_distance', 0),
            'policy_entropy': m.get('avg_policy_entropy', 0.0),
            'root_value': m.get('avg_root_value', 0.0),
            'mcts_depth': m.get('avg_mcts_depth', 0.0),
            'mcts_reward_std': m.get('avg_mcts_reward_std', 0.0),
            'mcts_reward_abs': m.get('avg_mcts_reward_abs', 0.0),
            'mcts_reward_sum_mean': m.get('avg_mcts_reward_sum_mean', 0.0),
            'mcts_boundary_frac': m.get('avg_mcts_boundary_frac', 0.0),
            'mcts_reached_terminal_frac': m.get('avg_mcts_reached_terminal_frac', 0.0),
            'mcts_sign_changes': m.get('avg_mcts_sign_changes', 0.0),
            'mcts_reward_frac': m.get('avg_mcts_reward_frac', 0.0),
            'game_time_s': m.get('game_time_s', 0.0),
            'num_simulations': g.get('num_simulations', 0),
            'weight_gen': g['weight_gen'],
            'weight_file': g.get('weight_file', ''),
            'weight_run_id': wl.get('run_id', '') or '',
            'weight_git_sha': wl.get('git_sha', '') or '',
            'selfplay_git_sha': sc.get('git_sha', '') or '',
            'slurm_job_id': sc.get('slurm_job_id', '') or '',
            'slurm_array_task_id': sc.get('slurm_array_task_id', '') or '',
            'reward_mode': g.get('reward_mode', ''),
            'prior_mix_weight': g.get('prior_mix_weight', 0.0),
            'batch_file': batch_file_rel,
            'game_idx': idx,
            'timestamp': ts,
        })
    return rows


def append_index(dataset_dir, game_dicts, batch_file_rel, node_id, batch_seq):
    rows = _index_rows_from_batch(game_dicts, batch_file_rel, node_id, batch_seq)
    table = pa.Table.from_pylist(rows, schema=INDEX_SCHEMA)
    index_dir = os.path.join(dataset_dir, 'index')
    os.makedirs(index_dir, exist_ok=True)
    shard_path = os.path.join(index_dir, f'{node_id}.parquet')
    if os.path.exists(shard_path):
        existing = pq.read_table(shard_path)
        table = pa.concat_tables([existing, table])
    pq.write_table(table, shard_path)


def query_index(dataset_dir, sql=None):
    import duckdb
    index_glob = os.path.join(dataset_dir, 'index', '*.parquet')
    conn = duckdb.connect()
    if sql is None:
        sql = f"SELECT * FROM read_parquet('{index_glob}')"
    elif not sql.strip().upper().startswith('SELECT'):
        # Treat as WHERE clause
        sql = f"SELECT * FROM read_parquet('{index_glob}') WHERE {sql}"
    else:
        sql = sql.replace('__INDEX__', f"read_parquet('{index_glob}')")
    return conn.execute(sql).fetchdf()


def load_selected_games(dataset_dir, batch_game_pairs):
    cache = {}
    games = []
    for batch_file, game_idx in batch_game_pairs:
        full_path = os.path.join(dataset_dir, batch_file)
        if full_path not in cache:
            cache[full_path] = torch.load(full_path, weights_only=False)
        games.append(cache[full_path][game_idx])
    return games


def dataset_stats(dataset_dir):
    batch_files = scan_dataset(dataset_dir)
    total_games = 0
    total_steps = 0
    class_counts = {}
    costs = []
    game_times = []
    entropies = []
    mcts_depths = []
    for path in batch_files:
        batch = torch.load(path, weights_only=False)
        parts = path.split(os.sep)
        mc = parts[-3] if len(parts) >= 3 else 'unknown'
        class_counts[mc] = class_counts.get(mc, 0) + len(batch)
        for g in batch:
            total_games += 1
            total_steps += len(g['history'])
            if 'cost' in g:
                costs.append(g['cost'])
            m = g.get('metrics', {})
            if 'game_time_s' in m:
                game_times.append(m['game_time_s'])
            if 'avg_policy_entropy' in m:
                entropies.append(m['avg_policy_entropy'])
            if 'avg_mcts_depth' in m:
                mcts_depths.append(m['avg_mcts_depth'])

    stats = {
        'total_games': total_games,
        'total_steps': total_steps,
        'total_batch_files': len(batch_files),
        'per_class': class_counts,
    }
    print(f"Dataset: {dataset_dir}")
    print(f"  {total_games} games, {total_steps} steps, {len(batch_files)} batch files")
    for mc, count in sorted(class_counts.items()):
        print(f"  {mc}: {count} games")
    if costs:
        stats['avg_cost'] = round(sum(costs) / len(costs), 1)
        stats['min_cost'] = min(costs)
        stats['max_cost'] = max(costs)
        print(f"  cost: avg={stats['avg_cost']}, min={stats['min_cost']}, max={stats['max_cost']}")
    if game_times:
        stats['avg_game_time_s'] = round(sum(game_times) / len(game_times), 2)
        print(f"  avg game time: {stats['avg_game_time_s']}s")
    if entropies:
        stats['avg_policy_entropy'] = round(sum(entropies) / len(entropies), 3)
        print(f"  avg policy entropy: {stats['avg_policy_entropy']}")
    if mcts_depths:
        stats['avg_mcts_depth'] = round(sum(mcts_depths) / len(mcts_depths), 1)
        print(f"  avg mcts depth: {stats['avg_mcts_depth']}")

    # Worker logs
    logs_dir = os.path.join(dataset_dir, 'logs')
    if os.path.isdir(logs_dir):
        summaries = [f for f in os.listdir(logs_dir) if f.endswith('_summary.json')]
        if summaries:
            total_worker_time = 0
            total_worker_games = 0
            for sf in summaries:
                with open(os.path.join(logs_dir, sf)) as f:
                    s = json.load(f)
                total_worker_time += s.get('total_time_s', 0)
                total_worker_games += s.get('total_games', 0)
            stats['num_workers_completed'] = len(summaries)
            stats['total_worker_cpu_hours'] = round(total_worker_time / 3600, 1)
            print(f"  workers: {len(summaries)} completed, "
                  f"{stats['total_worker_cpu_hours']} CPU-hours")

    return stats

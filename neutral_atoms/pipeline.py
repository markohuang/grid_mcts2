"""Pipeline registry helpers. Append-log is the source of truth; state.json is derived.

events.jsonl schema (one line per event):
  {"ts": "2026-04-21T13:45:00Z", "event": "pipeline_started", "pipeline_id": "abc12345", ...}

events we emit:
  pipeline_started           at kickoff
  cycle_NN_<phase>_launched  at sbatch time (phase = pretrain | selfplay | train)
  cycle_NN_<phase>_started   when the slurm job begins work
  cycle_NN_<phase>_completed on success
  cycle_NN_<phase>_failed    on error (best-effort)
  pipeline_completed         when final train ends
"""

import fcntl
import json
import os
import time


def _events_path(pipeline_dir):
    return os.path.join(pipeline_dir, 'events.jsonl')


def _state_path(pipeline_dir):
    return os.path.join(pipeline_dir, 'state.json')


def append_event(pipeline_dir, event, **fields):
    os.makedirs(pipeline_dir, exist_ok=True)
    entry = {'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'event': event, **fields}
    path = _events_path(pipeline_dir)
    with open(path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(entry) + '\n')
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def read_events(pipeline_dir):
    path = _events_path(pipeline_dir)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def derive_state(pipeline_dir):
    """Reduce events.jsonl into a snapshot. Pure function of the log."""
    events = read_events(pipeline_dir)
    state = {
        'pipeline_dir': pipeline_dir,
        'status': 'unknown',
        'current_cycle': None,
        'current_phase': None,
        'cycles': {},
    }
    for e in events:
        event = e['event']
        if event == 'pipeline_started':
            state['status'] = 'running'
            state['pipeline_id'] = e.get('pipeline_id')
            state['started_at'] = e['ts']
        elif event == 'pipeline_completed':
            state['status'] = 'completed'
            state['completed_at'] = e['ts']
            state['current_phase'] = None
        elif event.startswith('cycle_') or event.startswith('epoch_'):
            # Back-compat: old pipelines used 'epoch_' prefix.
            parts = event.split('_')
            idx = int(parts[1])
            phase = parts[2]
            outcome = '_'.join(parts[3:])
            cy = state['cycles'].setdefault(idx, {})
            cy.setdefault('phases', {}).setdefault(phase, {})[outcome] = e['ts']
            for k, v in e.items():
                if k in ('ts', 'event'):
                    continue
                cy['phases'][phase].setdefault('meta', {})[k] = v
            state['current_cycle'] = idx
            state['current_phase'] = phase
    return state


def write_state(pipeline_dir):
    """Atomically write state.json derived from events.jsonl."""
    state = derive_state(pipeline_dir)
    path = _state_path(pipeline_dir)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.rename(tmp, path)
    return state

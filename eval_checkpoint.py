import json
import time
from absl import app, flags
from ml_collections import config_flags

from neutral_atoms.config import get_config, atom_map_to_positions, set_derived_config, get_map_data
from neutral_atoms.network import Network
from neutral_atoms.game import Game
from neutral_atoms.mcts import play_game
from neutral_atoms.experiment import selfplay_metrics
from neutral_atoms.map_generator import generate_random_map

import torch


_CONFIG = config_flags.DEFINE_config_dict('config', get_config())
_CHECKPOINT = flags.DEFINE_string('checkpoint', '', 'Path to checkpoint file')
_NUM_GAMES = flags.DEFINE_integer('num_games', 100, 'Evaluation games per map')
_HOLDOUT_MAPS = flags.DEFINE_integer('holdout_maps', 4, 'Number of held-out random maps')
_HOLDOUT_SEED_START = flags.DEFINE_integer('holdout_seed_start', 10000, 'First held-out map seed')
_OUTPUT = flags.DEFINE_string('output', '', 'Optional JSON output path')
_DETERMINISTIC = flags.DEFINE_bool('deterministic', True, 'Disable root noise and choose argmax action')
_POLICY_ONLY = flags.DEFINE_bool('policy_only', False, 'Use num_simulations=1 for policy-only eval')


def _json_safe_metrics(metrics):
    return {k: v for k, v in metrics.items() if k != 'best_game'}


def _build_holdout_pool(config, tasks):
    gates_per_layer = max(len(layer) for layer in tasks)
    pool = []
    for offset in range(_HOLDOUT_MAPS.value):
        seed = _HOLDOUT_SEED_START.value + offset
        m = generate_random_map(
            (config.env.board_height, config.env.board_width),
            config.env.num_qubits,
            config.network.num_tasks,
            gates_per_layer=gates_per_layer,
            seed=seed,
        )
        pool.append({
            'seed': seed,
            'tasks': m['tasks'],
            'initial_positions': atom_map_to_positions(m['atom_map'], config.env.board_width),
        })
    return pool


def _eval_games(config, network, tasks, initial_positions):
    games = []
    for _ in range(_NUM_GAMES.value):
        game = Game(config, tasks, initial_positions)
        game = play_game(
            game, config.mcts, network,
            training_steps=network.training_steps(),
            add_exploration_noise=not _DETERMINISTIC.value,
            deterministic=_DETERMINISTIC.value,
        )
        games.append(game)
    return games


def main(_):
    config = _CONFIG.value
    set_derived_config(config)
    if _POLICY_ONLY.value:
        with config.mcts.unlocked():
            config.mcts.num_simulations = 1

    map_data = get_map_data(config)
    tasks = map_data['tasks']
    initial_positions = atom_map_to_positions(map_data['atom_map'], config.env.board_width)

    network = Network(config.network, use_fake=config.use_fake)
    if _CHECKPOINT.value:
        ckpt = torch.load(_CHECKPOINT.value, map_location='cpu')
        network.load_state_dict(ckpt['model'])
    network.eval()

    started_at = time.strftime('%Y-%m-%d %H:%M:%S')
    fixed_games = _eval_games(config, network, tasks, initial_positions)
    fixed_metrics = _json_safe_metrics(selfplay_metrics(fixed_games))

    holdout_pool = _build_holdout_pool(config, tasks)
    holdout_results = []
    holdout_all_games = []
    for entry in holdout_pool:
        games = _eval_games(config, network, entry['tasks'], entry['initial_positions'])
        holdout_all_games.extend(games)
        holdout_results.append({
            'seed': entry['seed'],
            'metrics': _json_safe_metrics(selfplay_metrics(games)),
        })
    holdout_metrics = _json_safe_metrics(selfplay_metrics(holdout_all_games)) if holdout_all_games else {}

    payload = {
        'started_at': started_at,
        'checkpoint': _CHECKPOINT.value,
        'map_num': config.map_num,
        'reward_mode': config.env.reward_mode,
        'num_games': _NUM_GAMES.value,
        'holdout_maps': _HOLDOUT_MAPS.value,
        'holdout_seed_start': _HOLDOUT_SEED_START.value,
        'deterministic': _DETERMINISTIC.value,
        'policy_only': _POLICY_ONLY.value,
        'num_simulations': config.mcts.num_simulations,
        'fixed_map': fixed_metrics,
        'holdout_aggregate': holdout_metrics,
        'holdout_maps_detail': holdout_results,
    }

    if _OUTPUT.value:
        with open(_OUTPUT.value, 'w') as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    app.run(main)

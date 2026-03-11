import os
from absl import app
from ml_collections import config_flags

from neutral_atoms.config import get_config, MAPS, atom_map_to_positions, set_derived_config
from neutral_atoms.network import Network
from neutral_atoms.trainer import AlphaAtomsTrainer

_CONFIG = config_flags.DEFINE_config_dict('config', get_config())


def print_config_summary(config, network):
    print(f"Map {config.map_num}: {config.env.board_height}x{config.env.board_width}, "
          f"{config.env.num_qubits} qubits, {config.network.num_tasks} tasks")
    print(f"Action space: {config.network.num_actions}, Use fake: {config.use_fake}")
    if not config.use_fake:
        n_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        print(f"Trainable params: {n_params}")


def main(_):
    config = _CONFIG.value
    set_derived_config(config)
    map_data = MAPS[config.map_num]
    tasks = map_data['tasks']
    initial_positions = atom_map_to_positions(map_data['atom_map'], config.env.board_width)

    network = Network(config.network, use_fake=config.use_fake)
    print_config_summary(config, network)

    trainer = AlphaAtomsTrainer(network, config, tasks, initial_positions)
    os.makedirs(config.training.save_dir, exist_ok=True)

    for epoch in range(config.training.epochs):
        print(f"\n=== Epoch {epoch+1}/{config.training.epochs} ===")
        print("Self-play:")
        trainer.run_selfplay()
        print("Training:")
        trainer.fit()


if __name__ == '__main__':
    app.run(main)

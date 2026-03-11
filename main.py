import os
from absl import app
from ml_collections import config_flags

from neutral_atoms.config import get_config, MAPS, atom_map_to_positions, set_derived_config
from neutral_atoms.network import Network
from neutral_atoms.trainer import AlphaAtomsTrainer
from neutral_atoms.experiment import (
    create_run_dir, selfplay_metrics, save_solution, append_to_registry,
)

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

    run_id, run_dir = create_run_dir(config)
    print(f"Run: {run_id} -> {run_dir}")

    trainer = AlphaAtomsTrainer(network, config, tasks, initial_positions)

    best_cost, best_game = float('inf'), None
    no_improve_count = 0
    run_metrics = {}
    epochs_completed = 0

    for epoch in range(config.training.epochs):
        epochs_completed = epoch + 1
        print(f"\n=== Epoch {epochs_completed}/{config.training.epochs} ===")

        print("Self-play:")
        games = trainer.run_selfplay()
        sp_metrics = selfplay_metrics(games, len(tasks))

        if 'best_cost' in sp_metrics:
            if sp_metrics['best_cost'] < best_cost:
                best_cost = sp_metrics['best_cost']
                best_game = sp_metrics['best_game']
                no_improve_count = 0
            else:
                no_improve_count += 1
            print(f"  best_cost={best_cost}, avg_cost={sp_metrics['avg_cost']:.1f}, "
                  f"completion={sp_metrics['completion_rate']:.0%}")
        else:
            print(f"  no completed games, completion={sp_metrics['completion_rate']:.0%}")

        print("Training:")
        train_result = trainer.fit()
        if train_result:
            run_metrics['final_loss'] = train_result['loss']

        if (epoch + 1) % config.experiment.checkpoint_every_n_epochs == 0:
            ckpt_path = os.path.join(run_dir, 'checkpoints', f'epoch_{epoch+1:03d}.ckpt')
            trainer.save_checkpoint(ckpt_path)
            print(f"  checkpoint: epoch_{epoch+1:03d}.ckpt")

        if no_improve_count >= config.experiment.early_stopping_patience:
            print(f"\nEarly stopping: no improvement for {no_improve_count} epochs")
            break

    # Save final artifacts
    trainer.save_checkpoint(os.path.join(run_dir, 'checkpoints', 'final.ckpt'))

    if best_game is not None:
        save_solution(best_game, os.path.join(run_dir, 'solutions', 'best.json'))
        print(f"\nBest solution cost: {best_cost}")

    run_metrics.update({
        'best_cost': best_cost if best_cost < float('inf') else None,
        'avg_cost': sp_metrics.get('avg_cost'),
        'completion_rate': sp_metrics.get('completion_rate', 0),
        'epochs_completed': epochs_completed,
    })
    append_to_registry(config.experiment.output_dir, run_id, config, run_metrics)
    print(f"Run {run_id} complete")


if __name__ == '__main__':
    app.run(main)

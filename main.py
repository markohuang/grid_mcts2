import torch
import os
import argparse

from neutral_atoms.types import EnvConfig
from neutral_atoms.config import MCTSConfig, TrainingConfig, NetworkConfig
from neutral_atoms.network import Network
from neutral_atoms.trainer import AlphaAtomsTrainer


def atom_map_to_positions(atom_map, board_width):
    """Convert flat-index atom map to (row, col) positions."""
    return [(idx // board_width, idx % board_width) for idx in atom_map]


MAPS = [
    {   # 2x6 with 9 atoms
        'board_dim': (2, 6), 'num_qubits': 9,
        'atom_map': [11, 3, 1, 8, 10, 7, 5, 6, 4],
        'tasks': [
            [[0, 8], [6, 2], [1, 7], [3, 5]],
            [[0, 2], [6, 8], [1, 5], [3, 7]],
            [[3, 7], [1, 5], [6, 8]],
        ],
    },
    {   # 4x4 with 8 atoms
        'board_dim': (4, 4), 'num_qubits': 8,
        'atom_map': [5, 9, 10, 11, 2, 6, 0, 14],
        'tasks': [
            [[0, 1], [2, 3], [5, 4], [6, 7]],
            [[0, 1], [2, 3], [5, 4], [6, 7]],
            [[0, 1], [2, 3], [5, 4], [6, 7]],
        ],
    },
    {   # 5x5 with 12 atoms
        'board_dim': (5, 5), 'num_qubits': 12,
        'atom_map': [4, 12, 7, 11, 15, 5, 2, 21, 22, 23, 20, 8],
        'tasks': [
            [[4, 3], [6, 7], [9, 8], [10, 11]],
            [[2, 1], [4, 5], [7, 6], [8, 9]],
            [[0, 1], [2, 3], [6, 5], [9, 8]],
        ],
    },
]


def main():
    parser = argparse.ArgumentParser(description='Alpha Atoms MCTS Training')
    parser.add_argument("--num_sims", type=int, default=50)
    parser.add_argument("--num_selfplay", type=int, default=2)
    parser.add_argument("--training_steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--map_num", type=int, default=0)
    parser.add_argument("--use_fake", action="store_true")
    parser.add_argument("--budget", type=int, default=24)
    args = parser.parse_args()

    m = MAPS[args.map_num]
    board_h, board_w = m['board_dim']
    num_qubits = m['num_qubits']
    board_size = board_h * board_w
    tasks = m['tasks']
    initial_positions = atom_map_to_positions(m['atom_map'], board_w)
    action_space_size = 1 + num_qubits * board_size

    env_config = EnvConfig(
        board_height=board_h, board_width=board_w,
        num_qubits=num_qubits, budget=args.budget,
    )
    mcts_config = MCTSConfig(num_simulations=args.num_sims)
    training_config = TrainingConfig(
        epochs=args.epochs, num_selfplay=args.num_selfplay,
        training_steps=args.training_steps, batch_size=args.batch_size,
        lr=args.lr,
    )
    net_config = NetworkConfig(
        num_tasks=len(tasks), num_qubits=num_qubits,
        board_size=board_size, num_actions=action_space_size,
    )

    network = Network(net_config, use_fake=args.use_fake)
    print(f"Map {args.map_num}: {board_h}x{board_w}, "
          f"{num_qubits} qubits, {len(tasks)} tasks")
    print(f"Action space: {action_space_size}, Use fake: {args.use_fake}")
    if not args.use_fake:
        n_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        print(f"Trainable params: {n_params}")

    trainer = AlphaAtomsTrainer(
        network, mcts_config, training_config, env_config,
        tasks, initial_positions, action_space_size,
    )

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        print("Self-play:")
        trainer.run_selfplay()
        print("Training:")
        trainer.fit()


if __name__ == '__main__':
    main()

"""
Generate template JSON for atom visualization.

Usage:
    python generate_template.py --rows 3 --cols 10 --qubits 9 --layers 5 --output public/data.json
    
    # Or with defaults:
    python generate_template.py
"""

import argparse
import json
import random
from pathlib import Path

def generate_random_positions(rows: int, cols: int, num_qubits: int) -> list[tuple[int, int]]:
    """Generate random unique positions for atoms."""
    all_positions = [(r, c) for r in range(rows) for c in range(cols)]
    if num_qubits > len(all_positions):
        raise ValueError(f"Cannot place {num_qubits} atoms in {rows}x{cols} grid")
    return random.sample(all_positions, num_qubits)

def generate_random_tasks(num_qubits: int, num_layers: int, gates_per_layer: int | None = None) -> list[list[list[int]]]:
    """
    Generate random gate layers.
    Each layer contains non-overlapping 2-qubit gates.
    """
    tasks = []
    for _ in range(num_layers):
        available = list(range(num_qubits))
        random.shuffle(available)
        
        # Determine number of gates for this layer
        max_gates = len(available) // 2
        if gates_per_layer is None:
            num_gates = random.randint(1, max(1, max_gates))
        else:
            num_gates = min(gates_per_layer, max_gates)
        
        layer = []
        for i in range(num_gates):
            if len(available) >= 2:
                a1 = available.pop()
                a2 = available.pop()
                layer.append([a1, a2])
        
        if layer:  # Only add non-empty layers
            tasks.append(layer)
    
    return tasks

def generate_template(
    rows: int = 3,
    cols: int = 10,
    num_qubits: int = 9,
    num_layers: int = 5,
    gates_per_layer: int | None = None,
    seed: int | None = None,
) -> dict:
    """Generate a complete template for visualization."""
    if seed is not None:
        random.seed(seed)
    
    # Generate random positions
    positions = generate_random_positions(rows, cols, num_qubits)
    
    # Generate random tasks
    circuit = generate_random_tasks(num_qubits, num_layers, gates_per_layer)
    
    # Build initial atoms dict
    initial_atoms = {
        str(i): {"row": pos[0], "col": pos[1]}
        for i, pos in enumerate(positions)
    }
    
    # Empty plan (baseline - no reconfig moves)
    plan = [[] for _ in circuit]
    
    return {
        "board": {
            "rows": rows,
            "cols": cols,
            "initialAtoms": initial_atoms
        },
        "circuit": circuit,
        "plan": plan
    }

def main():
    parser = argparse.ArgumentParser(description="Generate template JSON for atom visualization")
    parser.add_argument("--rows", type=int, default=3, help="Grid rows")
    parser.add_argument("--cols", type=int, default=10, help="Grid columns")
    parser.add_argument("--qubits", type=int, default=9, help="Number of qubits/atoms")
    parser.add_argument("--layers", type=int, default=5, help="Number of gate layers")
    parser.add_argument("--gates-per-layer", type=int, default=None, help="Gates per layer (random if not set)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="public/data.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Generate template
    data = generate_template(
        rows=args.rows,
        cols=args.cols,
        num_qubits=args.qubits,
        num_layers=args.layers,
        gates_per_layer=args.gates_per_layer,
        seed=args.seed,
    )
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated template:")
    print(f"  Grid: {args.rows} x {args.cols}")
    print(f"  Qubits: {args.qubits}")
    print(f"  Layers: {len(data['circuit'])}")
    print(f"  Circuit: {data['circuit']}")
    print(f"  Output: {output_path}")

if __name__ == "__main__":
    main()

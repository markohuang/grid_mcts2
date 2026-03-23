"""
Neutral Atoms Visualization API

Run with:
    uvicorn api:app --reload --port 8000

Or:
    python api.py

Requires neutral_atoms package in parent directory:
    project-root/
    ├── neutral_atoms/
    │   ├── __init__.py
    │   ├── moves.py
    │   └── ...
    └── atom-viz/
        └── api.py
"""

import sys
from pathlib import Path

# Add parent directory to path for neutral_atoms + baselines imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch

# Import from neutral_atoms package
from neutral_atoms.moves import parallel_groups, count_groups, canonicalize_moves

# Import baselines
from baselines.random_board import random_board as _random_board
from baselines.kohei_policy import plan as kohei_plan

app = FastAPI(title="Neutral Atoms Viz API")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API Models
# ============================================================================

class Position(BaseModel):
    row: int
    col: int

class MoveInput(BaseModel):
    atom: int
    frm: Position  # 'from' is reserved in Python
    to: Position

class GateInput(BaseModel):
    gates: list[list[int]]  # [[atom1, atom2], ...]
    atomPositions: dict[str, Position]  # {"0": {row, col}, ...}

class ReconfigInput(BaseModel):
    moves: list[MoveInput]

class ComputeRequest(BaseModel):
    reconfig: ReconfigInput | None = None
    gate: GateInput | None = None

class GroupResult(BaseModel):
    groups: list[int]  # group index for each move
    cost: int          # number of parallel steps (for reconfig) or 2x (for gates)
    numGroups: int     # raw number of groups

class ComputeResponse(BaseModel):
    reconfigResult: GroupResult | None = None
    gateResult: GroupResult | None = None
    totalCost: int

class LayerResult(BaseModel):
    layerIndex: int
    gates: list[list[int]]
    reconfigMoves: list[dict]
    reconfigGroups: list[int]
    reconfigCost: int
    gateGroups: list[int]
    gateCost: int
    atomPositions: dict[str, dict]

class ComputeAllResponse(BaseModel):
    layers: list[LayerResult]
    totalCost: int

class RandomBoardRequest(BaseModel):
    rows: int = 5
    cols: int = 5
    num_qubits: int = 12
    num_layers: int = 3
    gates_per_layer: int = 4
    seed: Optional[int] = None

class GeneratePlanRequest(BaseModel):
    board: dict
    circuit: list
    method: str  # "kohei" | "mcts"
    checkpoint_path: Optional[str] = None
    num_simulations: Optional[int] = 50

class GeneratePlanResponse(BaseModel):
    board: dict
    circuit: list
    plan: list
    method: str
    elapsed_ms: float

# ============================================================================
# Helper Functions
# ============================================================================

def moves_list_to_tensor(moves_list: list[list[int]]) -> torch.Tensor:
    """Convert list of [from_row, from_col, to_row, to_col] to tensor."""
    if not moves_list:
        return torch.empty(0, 4, dtype=torch.long)
    return torch.tensor(moves_list, dtype=torch.long)

def gates_to_moves_tensor(gates: list[list[int]], atom_positions: dict[str, dict]) -> torch.Tensor:
    """Convert gates to moves tensor: first atom moves to second atom's position."""
    if not gates:
        return torch.empty(0, 4, dtype=torch.long)
    moves_list = []
    for g in gates:
        atom1, atom2 = g
        pos1 = atom_positions[str(atom1)]
        pos2 = atom_positions[str(atom2)]
        moves_list.append([pos1["row"], pos1["col"], pos2["row"], pos2["col"]])
    return torch.tensor(moves_list, dtype=torch.long)

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "Neutral Atoms Viz API"}

@app.post("/compute", response_model=ComputeResponse)
def compute(request: ComputeRequest):
    """
    Compute parallel groups for reconfig moves and/or gate execution.
    """
    total_cost = 0
    reconfig_result = None
    gate_result = None
    
    # Process reconfig moves (canonicalize=False)
    if request.reconfig and len(request.reconfig.moves) > 0:
        moves_list = []
        for m in request.reconfig.moves:
            moves_list.append([m.frm.row, m.frm.col, m.to.row, m.to.col])
        moves_tensor = moves_list_to_tensor(moves_list)
        groups = parallel_groups(moves_tensor, canonicalize=False).tolist()
        num_groups = count_groups(moves_tensor, canonicalize=False)
        cost = num_groups  # reconfig cost = number of groups
        reconfig_result = GroupResult(groups=groups, cost=cost, numGroups=num_groups)
        total_cost += cost
    
    # Process gate execution (canonicalize=True)
    if request.gate and len(request.gate.gates) > 0:
        moves_tensor = gates_to_moves_tensor(request.gate.gates, request.gate.atomPositions)
        groups = parallel_groups(moves_tensor, canonicalize=True).tolist()
        num_groups = count_groups(moves_tensor, canonicalize=True)
        cost = 2 * num_groups  # gate cost = 2x (round trip)
        gate_result = GroupResult(groups=groups, cost=cost, numGroups=num_groups)
        total_cost += cost
    
    return ComputeResponse(
        reconfigResult=reconfig_result,
        gateResult=gate_result,
        totalCost=total_cost
    )

@app.post("/compute_all_layers")
def compute_all_layers(data: dict) -> dict:
    """
    Compute costs for all layers given board state and plan.
    Also validates moves and removes invalid ones.
    
    Expected input:
    {
        "board": {"rows": 3, "cols": 10, "initialAtoms": {...}},
        "circuit": [[[0,1], [2,3]], ...],
        "plan": [[], [{atom, from, to}, ...], ...]
    }
    """
    board = data["board"]
    circuit = data["circuit"]
    plan = data.get("plan", [[] for _ in circuit])
    
    # Ensure plan has same length as circuit
    while len(plan) < len(circuit):
        plan.append([])
    
    # Build initial atom positions
    atom_positions: dict[int, dict] = {}
    for atom_id, pos in board["initialAtoms"].items():
        atom_positions[int(atom_id)] = {"row": pos["row"], "col": pos["col"]}
    
    results = []
    total_cost = 0
    validated_plan = []  # Plan after validation
    
    for layer_idx, gates in enumerate(circuit):
        layer_moves = plan[layer_idx] if layer_idx < len(plan) else []
        
        # Validate moves for this layer
        valid_moves = []
        planned_destinations = set()  # Track destinations to prevent collisions
        atoms_being_moved = set()  # Track which atoms have moves
        
        for m in layer_moves:
            atom_id = m["atom"]
            expected_from = atom_positions.get(atom_id)
            move_from = m.get("from", expected_from)
            move_to = m["to"]
            
            # Check 1: Move starts from atom's current position
            if expected_from is None:
                continue
            if move_from["row"] != expected_from["row"] or move_from["col"] != expected_from["col"]:
                continue  # Invalid: atom is not at expected position
            
            # Check 2: Destination not already planned by another move in this layer
            dest_key = (move_to["row"], move_to["col"])
            if dest_key in planned_destinations:
                continue  # Invalid: collision with another planned move
            
            # Check 3: Destination not occupied by unmoved atom
            is_occupied = False
            for other_id, other_pos in atom_positions.items():
                if other_id == atom_id:
                    continue
                other_at_dest = (other_pos["row"] == move_to["row"] and 
                                other_pos["col"] == move_to["col"])
                # Check if other atom is being moved away from this position
                other_being_moved_away = False
                for vm in valid_moves:
                    if vm["atom"] == other_id:
                        other_being_moved_away = True
                        break
                
                if other_at_dest and not other_being_moved_away:
                    is_occupied = True
                    break
            
            if is_occupied:
                continue  # Invalid: destination occupied by unmoved atom
            
            # Move is valid
            valid_moves.append({
                "atom": atom_id,
                "from": {"row": expected_from["row"], "col": expected_from["col"]},
                "to": {"row": move_to["row"], "col": move_to["col"]}
            })
            planned_destinations.add(dest_key)
            atoms_being_moved.add(atom_id)
        
        validated_plan.append(valid_moves)
        
        # Compute reconfig cost
        layer_result = {
            "layerIndex": layer_idx,
            "gates": gates,
            "canonicalizedGates": gates,  # Default to original, updated below if gates exist
            "reconfigMoves": valid_moves,
            "reconfigGroups": [],
            "reconfigCost": 0,
            "gateGroups": [],
            "gateCost": 0,
            "atomPositions": {},
        }
        
        if valid_moves:
            moves_list = [[m["from"]["row"], m["from"]["col"], m["to"]["row"], m["to"]["col"]] 
                         for m in valid_moves]
            moves_tensor = moves_list_to_tensor(moves_list)
            groups = parallel_groups(moves_tensor, canonicalize=False).tolist()
            cost = count_groups(moves_tensor, canonicalize=False)
            layer_result["reconfigGroups"] = groups
            layer_result["reconfigCost"] = cost
            total_cost += cost
            
            # Apply moves to atom positions
            for m in valid_moves:
                atom_positions[m["atom"]] = {"row": m["to"]["row"], "col": m["to"]["col"]}
        
        # Compute gate execution cost
        if gates:
            # Convert atom_positions to string keys for helper
            pos_str_keys = {str(k): v for k, v in atom_positions.items()}
            moves_tensor = gates_to_moves_tensor(gates, pos_str_keys)
            
            # Canonicalize the moves for optimal grouping
            canonicalized_tensor = canonicalize_moves(moves_tensor)
            groups = parallel_groups(canonicalized_tensor, canonicalize=False).tolist()
            num_groups = count_groups(canonicalized_tensor, canonicalize=False)
            cost = 2 * num_groups
            layer_result["gateGroups"] = groups
            layer_result["gateCost"] = cost
            total_cost += cost
            
            # Convert canonicalized moves back to gate format for frontend
            # Find which atom is at each position
            pos_to_atom = {(v["row"], v["col"]): k for k, v in atom_positions.items()}
            canonicalized_gates = []
            for i in range(len(gates)):
                move = canonicalized_tensor[i].tolist()
                from_pos = (move[0], move[1])
                to_pos = (move[2], move[3])
                from_atom = pos_to_atom.get(from_pos)
                to_atom = pos_to_atom.get(to_pos)
                if from_atom is not None and to_atom is not None:
                    canonicalized_gates.append([from_atom, to_atom])
                else:
                    # Fallback to original gate
                    canonicalized_gates.append(gates[i])
            layer_result["canonicalizedGates"] = canonicalized_gates
        
        # Store current atom positions for this layer (AFTER reconfig applied)
        layer_result["atomPositions"] = {str(k): {"row": v["row"], "col": v["col"]} for k, v in atom_positions.items()}
        results.append(layer_result)
    
    return {
        "layers": results,
        "totalCost": total_cost,
        "validatedPlan": validated_plan  # Return validated plan for frontend to sync
    }

# ============================================================================
# Baseline endpoints
# ============================================================================

@app.post("/random_board")
def random_board_endpoint(request: RandomBoardRequest) -> dict:
    """Generate a random board + circuit in atom-viz format."""
    return _random_board(
        rows=request.rows,
        cols=request.cols,
        num_qubits=request.num_qubits,
        num_layers=request.num_layers,
        gates_per_layer=request.gates_per_layer,
        seed=request.seed,
    )

@app.post("/generate_plan", response_model=GeneratePlanResponse)
def generate_plan(request: GeneratePlanRequest) -> dict:
    """Generate a plan using the specified method (kohei | dpqa | mcts)."""
    import time

    board = request.board
    circuit = request.circuit
    rows, cols = board["rows"], board["cols"]
    initial = {int(k): (v["row"], v["col"])
               for k, v in board["initialAtoms"].items()}
    tasks = [[(g[0], g[1]) for g in layer] for layer in circuit]

    t0 = time.time()

    if request.method == "kohei":
        result = kohei_plan(initial, tasks, rows, cols)
        return GeneratePlanResponse(
            board=result["board"],
            circuit=result["circuit"],
            plan=result["plan"],
            method="kohei",
            elapsed_ms=(time.time() - t0) * 1000,
        )

    elif request.method == "mcts":
        raise HTTPException(status_code=501,
                            detail="MCTS inference not yet available — train a model first.")

    else:
        raise HTTPException(status_code=400,
                            detail=f"Unknown method '{request.method}'. Use: kohei, mcts.")

# ============================================================================
# Run server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

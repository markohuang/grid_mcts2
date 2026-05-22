"""Standalone MCTS inference service.

Run from the repository root:

    uvicorn inference_service.api:app --host 0.0.0.0 --port 8001

The atom-viz gateway calls this service over HTTP. Do not import this module
from atom-viz/api.py; that would couple the lightweight gateway to the GPU
runtime.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from fast_mcts.backends import available as available_backends
from fast_mcts.backends import get_backend
from neutral_atoms.config import set_derived_config
from neutral_atoms.config_hpc import get_config
from neutral_atoms.experiment import compute_solution_cost, game_to_solution
from neutral_atoms.game import Game
from neutral_atoms.network import Network


DEFAULT_MODEL_ID = "v3a01-cycle05"
DEFAULT_CHECKPOINT = "pipelines/pipeline_v3a01/checkpoints/cycle_05.ckpt"


class Position(BaseModel):
    row: int
    col: int


class Board(BaseModel):
    rows: int
    cols: int
    initialAtoms: dict[str, Position]


class PlanRequest(BaseModel):
    board: Board
    circuit: list[list[list[int]]]
    model_id: str = DEFAULT_MODEL_ID
    checkpoint_path: Optional[str] = None
    num_simulations: int = Field(default=400, ge=1)
    backend: str = "fast"
    device: str = "auto"
    deterministic: bool = True
    map_num: int = 2


class ModelInfo(BaseModel):
    id: str
    checkpoint_path: str
    run_id: Optional[str] = None
    training_steps: Optional[int] = None
    board: str
    num_qubits: int
    num_layers: int
    default_backend: str
    default_num_simulations: int


class PlanResponse(BaseModel):
    board: dict
    circuit: list
    plan: list
    model_id: str
    checkpoint_run_id: Optional[str] = None
    training_steps: Optional[int] = None
    plan_cost: int
    elapsed_ms: float
    device: str
    backend: str
    num_simulations: int
    diagnostics: dict


app = FastAPI(title="Neutral Atoms MCTS Inference Service")

_MODEL_CACHE: dict[tuple[str, str, str], tuple[Network, dict]] = {}


def _service_token() -> str:
    return os.environ.get("MCTS_INFERENCE_TOKEN", "")


def _check_auth(authorization: Optional[str]) -> None:
    token = _service_token()
    if not token:
        return
    expected = f"Bearer {token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid inference service token.")


def _model_registry() -> dict[str, str]:
    return {
        DEFAULT_MODEL_ID: os.environ.get("MCTS_DEFAULT_CHECKPOINT", DEFAULT_CHECKPOINT),
    }


def _resolve_checkpoint(model_id: str, checkpoint_path: Optional[str]) -> str:
    allow_path = os.environ.get("MCTS_ALLOW_CHECKPOINT_PATH", "0") == "1"
    if checkpoint_path:
        if not allow_path:
            raise HTTPException(
                status_code=403,
                detail="Raw checkpoint_path is disabled. Use a registered model_id.",
            )
        return checkpoint_path

    registry = _model_registry()
    if model_id not in registry:
        raise HTTPException(status_code=404, detail=f"Unknown model_id '{model_id}'.")
    return registry[model_id]


def _select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="CUDA was requested but is not available.")
    if requested not in ("cpu", "cuda"):
        raise HTTPException(status_code=422, detail="device must be one of: auto, cpu, cuda.")
    return requested


def _build_config(req: PlanRequest):
    cfg = get_config()
    cfg.map_num = req.map_num
    set_derived_config(cfg)

    num_qubits = len(req.board.initialAtoms)
    num_layers = len(req.circuit)
    if req.board.rows != cfg.env.board_height or req.board.cols != cfg.env.board_width:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Model '{req.model_id}' expects {cfg.env.board_height}x{cfg.env.board_width} "
                f"boards, got {req.board.rows}x{req.board.cols}."
            ),
        )
    if num_qubits != cfg.env.num_qubits:
        raise HTTPException(
            status_code=422,
            detail=f"Model '{req.model_id}' expects {cfg.env.num_qubits} qubits, got {num_qubits}.",
        )
    if num_layers != cfg.network.num_tasks:
        raise HTTPException(
            status_code=422,
            detail=f"Model '{req.model_id}' expects {cfg.network.num_tasks} layers, got {num_layers}.",
        )
    if req.backend not in available_backends():
        raise HTTPException(
            status_code=422,
            detail=f"Unknown backend '{req.backend}'. Available: {available_backends()}",
        )

    max_sims = int(os.environ.get("MCTS_MAX_NUM_SIMULATIONS", "10000"))
    if req.num_simulations > max_sims:
        raise HTTPException(
            status_code=413,
            detail=f"num_simulations={req.num_simulations} exceeds limit {max_sims}.",
        )

    with cfg.mcts.unlocked():
        cfg.mcts.num_simulations = req.num_simulations
        cfg.mcts.backend = req.backend
    return cfg


def _tasks_from_circuit(circuit: list[list[list[int]]], num_qubits: int) -> list[list[list[int]]]:
    tasks = []
    for layer_idx, layer in enumerate(circuit):
        seen = set()
        out_layer = []
        for gate in layer:
            if len(gate) != 2:
                raise HTTPException(status_code=422, detail=f"Layer {layer_idx} contains a non-pair gate.")
            a, b = int(gate[0]), int(gate[1])
            if a == b or a < 0 or b < 0 or a >= num_qubits or b >= num_qubits:
                raise HTTPException(status_code=422, detail=f"Invalid gate [{a}, {b}] in layer {layer_idx}.")
            if a in seen or b in seen:
                raise HTTPException(status_code=422, detail=f"Qubit appears in multiple gates in layer {layer_idx}.")
            seen.add(a)
            seen.add(b)
            out_layer.append([a, b])
        tasks.append(out_layer)
    return tasks


def _initial_positions(board: Board, num_qubits: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int] | None] = [None] * num_qubits
    occupied = set()
    for raw_id, pos in board.initialAtoms.items():
        try:
            atom_id = int(raw_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid atom id '{raw_id}'.") from exc
        if atom_id < 0 or atom_id >= num_qubits:
            raise HTTPException(status_code=422, detail=f"Atom id {atom_id} is out of range.")
        if pos.row < 0 or pos.row >= board.rows or pos.col < 0 or pos.col >= board.cols:
            raise HTTPException(status_code=422, detail=f"Atom {atom_id} position is outside the board.")
        key = (pos.row, pos.col)
        if key in occupied:
            raise HTTPException(status_code=422, detail="Two atoms occupy the same cell.")
        occupied.add(key)
        positions[atom_id] = key
    missing = [i for i, p in enumerate(positions) if p is None]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing atom positions for ids: {missing}.")
    return [p for p in positions if p is not None]


def _load_model(model_id: str, checkpoint_path: str, device: str, cfg) -> tuple[Network, dict]:
    abs_path = str(Path(checkpoint_path).expanduser().resolve())
    if not Path(abs_path).exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found for model '{model_id}'.")
    cache_key = (model_id, abs_path, device)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    net = Network(cfg.network, use_fake=False)
    ckpt = torch.load(abs_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    _MODEL_CACHE[cache_key] = (net, ckpt if isinstance(ckpt, dict) else {})
    return _MODEL_CACHE[cache_key]


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "neutral-atoms-mcts-inference",
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.get("/v1/models")
def models(authorization: Optional[str] = Header(default=None)) -> dict:
    _check_auth(authorization)
    registry = _model_registry()
    result = []
    for model_id, ckpt_path in registry.items():
        info = {
            "id": model_id,
            "checkpoint_path": ckpt_path,
            "board": "5x5",
            "num_qubits": 12,
            "num_layers": 3,
            "default_backend": "fast",
            "default_num_simulations": 400,
        }
        path = Path(ckpt_path)
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                info["run_id"] = ckpt.get("run_id")
                info["training_steps"] = ckpt.get("training_steps")
        result.append(info)
    return {"models": result}


@app.post("/v1/plan", response_model=PlanResponse)
def plan(req: PlanRequest, authorization: Optional[str] = Header(default=None)) -> PlanResponse:
    _check_auth(authorization)
    started = time.time()
    checkpoint_path = _resolve_checkpoint(req.model_id, req.checkpoint_path)
    device = _select_device(req.device)
    cfg = _build_config(req)
    tasks = _tasks_from_circuit(req.circuit, cfg.env.num_qubits)
    initial_positions = _initial_positions(req.board, cfg.env.num_qubits)

    net, ckpt = _load_model(req.model_id, checkpoint_path, device, cfg)
    game = Game(cfg, tasks, initial_positions)

    play = get_backend(req.backend)
    with torch.no_grad():
        game = play(
            game,
            cfg.mcts,
            net,
            add_exploration_noise=not req.deterministic,
            deterministic=req.deterministic,
        )

    solution = game_to_solution(game)
    cost = compute_solution_cost(game)
    elapsed_ms = (time.time() - started) * 1000
    return PlanResponse(
        board=solution["board"],
        circuit=solution["circuit"],
        plan=solution["plan"],
        model_id=req.model_id,
        checkpoint_run_id=ckpt.get("run_id"),
        training_steps=ckpt.get("training_steps"),
        plan_cost=cost,
        elapsed_ms=elapsed_ms,
        device=device,
        backend=req.backend,
        num_simulations=req.num_simulations,
        diagnostics={
            "avg_mcts_depth": (
                sum(game.mcts_depths) / len(game.mcts_depths)
                if game.mcts_depths else None
            ),
            "root_value_mean": (
                sum(game.root_values) / len(game.root_values)
                if game.root_values else None
            ),
        },
    )

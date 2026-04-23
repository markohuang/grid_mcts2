"""Backend registry.

A backend is any callable with signature
    play_game(game, mcts_config, network, *, add_exploration_noise, deterministic,
              temperature_override) -> Game

`classic` wraps `neutral_atoms.mcts.play_game` unchanged. Future `fast`
backend will register here. Harnesses in this package accept a backend name
string and dispatch — classic/fast comparison goes through the same code path.
"""

from __future__ import annotations
from typing import Callable


CLASSIC_BACKEND = 'classic'
_REGISTRY: dict[str, Callable] = {}


def register(name: str, fn: Callable) -> None:
    _REGISTRY[name] = fn


def get_backend(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend '{name}'. available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


FAST_BACKEND = 'fast'


def _register_classic():
    from neutral_atoms.mcts import play_game
    register(CLASSIC_BACKEND, play_game)


def _register_fast():
    from .search import play_game
    register(FAST_BACKEND, play_game)


_register_classic()
_register_fast()

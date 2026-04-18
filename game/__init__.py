# Game registry: map game name -> module path so tournament.py and main.py
# don't need to know about specific games.
from importlib import import_module
from types import ModuleType

_GAMES = {
    "checkers": "game.checkers",
    "othello": "game.othello",
    "tictactoe": "game.tictactoe",
}


def get_game(name: str) -> ModuleType:
    """Return the game module for a given name."""
    key = name.lower()
    if key not in _GAMES:
        raise ValueError(
            f"unknown game: {name!r}; available: {sorted(_GAMES)}"
        )
    return import_module(_GAMES[key])


def available_games() -> list[str]:
    return sorted(_GAMES)

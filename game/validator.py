"""Move validator: dispatches to the correct game engine by name."""

from __future__ import annotations


def validate(move_str: str, state, player: int, game: str = "checkers") -> bool:
    if not move_str:
        return False
    if game == "checkers":
        from game.checkers import get_legal_moves
    elif game == "tictactoe":
        from game.tictactoe import get_legal_moves
    else:
        raise ValueError(f"unknown game: {game!r}")
    return move_str.strip() in get_legal_moves(state, player)

"""
Tic-tac-toe game logic, matching the checkers module interface.

Board: numpy int8 3x3. 0 empty, 1 X (player 1), 2 O (player 2).
Notation: files a-c left to right, ranks 1-3 bottom to top (rank 1 = row index 2).
Single-square moves like "a1", "b2", "c3" (no hyphens).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

Coord = Tuple[int, int]

EMPTY = 0
X_MARK = 1  # player 1
O_MARK = 2  # player 2

_LINES: List[List[Coord]] = [
    # rows
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    # columns
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
    # diagonals
    [(0, 0), (1, 1), (2, 2)],
    [(0, 2), (1, 1), (2, 0)],
]


def initial_state() -> np.ndarray:
    return np.zeros((3, 3), dtype=np.int8)


def parse_square(s: str) -> Coord:
    s = s.strip().lower()
    if len(s) != 2:
        raise ValueError(f"bad square: {s!r}")
    f, rk = s[0], s[1]
    if f < "a" or f > "c" or rk < "1" or rk > "3":
        raise ValueError(f"bad square: {s!r}")
    col = ord(f) - ord("a")
    row = 3 - int(rk)
    return row, col


def format_square(r: int, c: int) -> str:
    return f"{chr(ord('a') + c)}{3 - r}"


def _owner(piece: int) -> Optional[int]:
    if piece == X_MARK:
        return 1
    if piece == O_MARK:
        return 2
    return None


def get_legal_moves(state: np.ndarray, player: int) -> List[str]:
    b = np.asarray(state, dtype=np.int8)
    if player not in (1, 2):
        raise ValueError("player must be 1 (X) or 2 (O)")
    moves: List[str] = []
    for r in range(3):
        for c in range(3):
            if int(b[r, c]) == EMPTY:
                moves.append(format_square(r, c))
    return sorted(moves)


def apply_move(state: np.ndarray, move_str: str, player: int) -> np.ndarray:
    b = np.asarray(state, dtype=np.int8).copy()
    if player not in (1, 2):
        raise ValueError("player must be 1 (X) or 2 (O)")
    r, c = parse_square(move_str)
    if int(b[r, c]) != EMPTY:
        raise ValueError("destination occupied")
    b[r, c] = X_MARK if player == 1 else O_MARK
    return b


def _winner(b: np.ndarray) -> Optional[int]:
    for line in _LINES:
        vals = [int(b[r, c]) for r, c in line]
        if vals[0] != EMPTY and vals[0] == vals[1] == vals[2]:
            return _owner(vals[0])
    return None


def is_terminal(state: np.ndarray, player: int) -> Tuple[bool, Optional[int]]:
    """
    Terminal if someone has a 3-in-a-row, or the board is full (draw).
    Returns (True, winner) where winner is 1, 2, or None for a draw.
    Unlike checkers, draws ARE inferred from the board alone.
    The `player` argument is accepted for API compatibility but not used.
    """
    _ = player
    b = np.asarray(state, dtype=np.int8)
    w = _winner(b)
    if w is not None:
        return (True, w)
    if not np.any(b == EMPTY):
        return (True, None)
    return (False, None)


def board_to_ascii(state: np.ndarray) -> str:
    b = np.asarray(state, dtype=np.int8)
    sym = {EMPTY: ".", X_MARK: "X", O_MARK: "O"}
    lines: List[str] = []
    lines.append("   a b c")
    for r in range(3):
        rank = 3 - r
        row_chars = [sym[int(b[r, c])] for c in range(3)]
        lines.append(f"{rank}  " + " ".join(row_chars) + f"  {rank}")
    lines.append("   a b c")
    return "\n".join(lines)

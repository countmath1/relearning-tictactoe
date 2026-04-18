"""
Othello (Reversi) game logic, matching the checkers/tictactoe module interface.

Board: numpy int8 8x8. 0 empty, 1 black, 2 white.
Black moves first.
Notation: files a-h left to right, ranks 1-8 bottom to top (rank 1 = row index 7).
Single-square placement moves like "d3", "c4" (no hyphens).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

Coord = Tuple[int, int]

EMPTY = 0
BLACK = 1  # player 1
WHITE = 2  # player 2

# Eight directions: (dr, dc)
_DIRS: List[Coord] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def initial_state() -> np.ndarray:
    b = np.zeros((8, 8), dtype=np.int8)
    # Standard Othello starting position: center 2x2
    b[3, 3] = WHITE
    b[3, 4] = BLACK
    b[4, 3] = BLACK
    b[4, 4] = WHITE
    return b


def parse_square(s: str) -> Coord:
    s = s.strip().lower()
    if len(s) != 2:
        raise ValueError(f"bad square: {s!r}")
    f, rk = s[0], s[1]
    if f < "a" or f > "h" or rk < "1" or rk > "8":
        raise ValueError(f"bad square: {s!r}")
    col = ord(f) - ord("a")
    row = 8 - int(rk)
    return row, col


def format_square(r: int, c: int) -> str:
    return f"{chr(ord('a') + c)}{8 - r}"


def _opponent(player: int) -> int:
    return 2 if player == 1 else 1


def _flips_in_dir(
    board: np.ndarray, r: int, c: int, dr: int, dc: int, player: int
) -> List[Coord]:
    """Return list of opponent squares that would be flipped in one direction."""
    opp = _opponent(player)
    flipped: List[Coord] = []
    cr, cc = r + dr, c + dc
    while 0 <= cr < 8 and 0 <= cc < 8:
        val = int(board[cr, cc])
        if val == opp:
            flipped.append((cr, cc))
        elif val == player:
            return flipped  # bracketed: these all flip
        else:
            return []  # hit empty: no bracket
        cr += dr
        cc += dc
    return []  # ran off edge: no bracket


def _all_flips(board: np.ndarray, r: int, c: int, player: int) -> List[Coord]:
    """All squares flipped by placing player's disc at (r, c)."""
    if int(board[r, c]) != EMPTY:
        return []
    result: List[Coord] = []
    for dr, dc in _DIRS:
        result.extend(_flips_in_dir(board, r, c, dr, dc, player))
    return result


def get_legal_moves(state: np.ndarray, player: int) -> List[str]:
    b = np.asarray(state, dtype=np.int8)
    if player not in (1, 2):
        raise ValueError("player must be 1 (black) or 2 (white)")
    moves: List[str] = []
    for r in range(8):
        for c in range(8):
            if _all_flips(b, r, c, player):
                moves.append(format_square(r, c))
    return sorted(moves)


def apply_move(state: np.ndarray, move_str: str, player: int) -> np.ndarray:
    b = np.asarray(state, dtype=np.int8).copy()
    if player not in (1, 2):
        raise ValueError("player must be 1 (black) or 2 (white)")
    r, c = parse_square(move_str)
    flips = _all_flips(b, r, c, player)
    if not flips:
        raise ValueError(f"illegal move: {move_str} does not flip any discs")
    b[r, c] = player
    for fr, fc in flips:
        b[fr, fc] = player
    return b


def is_terminal(state: np.ndarray, player: int) -> Tuple[bool, Optional[int]]:
    """
    Terminal when neither player can move.
    Winner is the player with more discs; None for a tie.
    """
    b = np.asarray(state, dtype=np.int8)
    if get_legal_moves(b, 1) or get_legal_moves(b, 2):
        return (False, None)
    # Game over: count discs
    n_black = int(np.sum(b == BLACK))
    n_white = int(np.sum(b == WHITE))
    if n_black > n_white:
        return (True, 1)
    if n_white > n_black:
        return (True, 2)
    return (True, None)  # tie


def board_to_ascii(state: np.ndarray) -> str:
    b = np.asarray(state, dtype=np.int8)
    sym = {EMPTY: ".", BLACK: "b", WHITE: "w"}
    lines: List[str] = []
    lines.append("   a b c d e f g h")
    for r in range(8):
        rank = 8 - r
        row_chars = [sym[int(b[r, c])] for c in range(8)]
        lines.append(f"{rank}  " + " ".join(row_chars) + f"  {rank}")
    lines.append("   a b c d e f g h")
    return "\n".join(lines)

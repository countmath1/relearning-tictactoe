"""
American checkers (English draughts) game logic.

Board: numpy int8 8x8. 0 empty, 1 black man, 2 red man, 3 black king, 4 red king.
Black starts at top (row 0), red at bottom (row 7).
Notation: files a-h left to right, ranks 1-8 bottom to top (rank 1 = row index 7).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

Coord = Tuple[int, int]

EMPTY = 0
BLACK_MAN = 1
RED_MAN = 2
BLACK_KING = 3
RED_KING = 4


def initial_state() -> np.ndarray:
    b = np.zeros((8, 8), dtype=np.int8)
    for r in range(3):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r, c] = BLACK_MAN
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                b[r, c] = RED_MAN
    return b


def _is_dark(r: int, c: int) -> bool:
    return (r + c) % 2 == 1


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


def _owner(piece: int) -> Optional[int]:
    if piece in (BLACK_MAN, BLACK_KING):
        return 1
    if piece in (RED_MAN, RED_KING):
        return 2
    return None


def _is_king(piece: int) -> bool:
    return piece in (BLACK_KING, RED_KING)


def _forward_dr(player: int) -> int:
    return 1 if player == 1 else -1


def _maybe_promote(board: np.ndarray, r: int, c: int) -> None:
    p = int(board[r, c])
    if p == BLACK_MAN and r == 7:
        board[r, c] = BLACK_KING
    elif p == RED_MAN and r == 0:
        board[r, c] = RED_KING


def _capture_dirs(board: np.ndarray, r: int, c: int, player: int) -> List[Tuple[int, int]]:
    """Unit diagonal steps allowed for captures from (r,c)."""
    p = int(board[r, c])
    if p == EMPTY:
        return []
    dirs: List[Tuple[int, int]] = []
    if _is_king(p):
        for dr in (-1, 1):
            for dc in (-1, 1):
                dirs.append((dr, dc))
    else:
        dr = _forward_dr(player)
        dirs.append((dr, -1))
        dirs.append((dr, 1))
    return dirs


def _step_dirs(board: np.ndarray, r: int, c: int, player: int) -> List[Tuple[int, int]]:
    """Unit diagonal steps for non-capture moves."""
    return _capture_dirs(board, r, c, player)


def _enemy_of(player: int, piece: int) -> bool:
    o = _owner(piece)
    return o is not None and o != player


def _single_capture_landings(
    board: np.ndarray, r: int, c: int, player: int
) -> List[Coord]:
    out: List[Coord] = []
    p = int(board[r, c])
    if p == EMPTY or _owner(p) != player:
        return out
    for dr, dc in _capture_dirs(board, r, c, player):
        mr, mc = r + dr, c + dc
        nr, nc = r + 2 * dr, c + 2 * dc
        if not (0 <= nr < 8 and 0 <= nc < 8 and 0 <= mr < 8 and 0 <= mc < 8):
            continue
        if not _enemy_of(player, int(board[mr, mc])):
            continue
        if int(board[nr, nc]) != EMPTY:
            continue
        out.append((nr, nc))
    return out


def _any_capture_from_board(board: np.ndarray, player: int) -> bool:
    for r in range(8):
        for c in range(8):
            if _owner(int(board[r, c])) != player:
                continue
            if _single_capture_landings(board, r, c, player):
                return True
    return False


def _apply_one_jump(board: np.ndarray, r: int, c: int, nr: int, nc: int, player: int) -> None:
    dr = (nr - r) // 2
    dc = (nc - c) // 2
    mr, mc = r + dr, c + dc
    jumped = int(board[mr, mc])
    if not _enemy_of(player, jumped):
        raise ValueError("invalid jump")
    piece = int(board[r, c])
    board[r, c] = EMPTY
    board[mr, mc] = EMPTY
    board[nr, nc] = piece
    _maybe_promote(board, nr, nc)


def _simple_step_landings(board: np.ndarray, r: int, c: int, player: int) -> List[Coord]:
    out: List[Coord] = []
    for dr, dc in _step_dirs(board, r, c, player):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 8 and 0 <= nc < 8 and int(board[nr, nc]) == EMPTY:
            out.append((nr, nc))
    return out


def _path_to_str(squares: Sequence[Coord]) -> str:
    parts = [format_square(r, c) for r, c in squares]
    return "-".join(parts)


def _enumerate_capture_sequences(board: np.ndarray, player: int) -> List[str]:
    """All full mandatory capture sequences for `player` (multi-jump until done)."""
    results: List[str] = []

    def dfs(brd: np.ndarray, r: int, c: int, path: List[Coord]) -> None:
        lands = _single_capture_landings(brd, r, c, player)
        if not lands:
            if len(path) >= 2:
                results.append(_path_to_str(path))
            return
        for nr, nc in lands:
            nb = brd.copy()
            _apply_one_jump(nb, r, c, nr, nc, player)
            dfs(nb, nr, nc, path + [(nr, nc)])

    for r in range(8):
        for c in range(8):
            if _owner(int(board[r, c])) != player:
                continue
            lands = _single_capture_landings(board, r, c, player)
            for nr, nc in lands:
                nb = board.copy()
                _apply_one_jump(nb, r, c, nr, nc, player)
                dfs(nb, nr, nc, [(r, c), (nr, nc)])

    # Deduplicate while keeping deterministic order
    seen: Set[str] = set()
    uniq: List[str] = []
    for s in sorted(results):
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def get_legal_moves(state: np.ndarray, player: int) -> List[str]:
    b = np.asarray(state, dtype=np.int8)
    if player not in (1, 2):
        raise ValueError("player must be 1 (black) or 2 (red)")

    caps = _enumerate_capture_sequences(b, player)
    if caps:
        return caps

    moves: List[str] = []
    for r in range(8):
        for c in range(8):
            if _owner(int(b[r, c])) != player:
                continue
            for nr, nc in _simple_step_landings(b, r, c, player):
                moves.append(f"{format_square(r, c)}-{format_square(nr, nc)}")
    return sorted(moves)


def apply_move(state: np.ndarray, move_str: str, player: int) -> np.ndarray:
    b = np.asarray(state, dtype=np.int8).copy()
    parts = [p.strip() for p in move_str.strip().split("-") if p.strip()]
    if len(parts) < 2:
        raise ValueError("move must have at least two squares")
    coords = [parse_square(p) for p in parts]

    sr, sc = coords[0]
    if _owner(int(b[sr, sc])) != player:
        raise ValueError("wrong piece at start")

    dr0 = coords[1][0] - sr
    dc0 = coords[1][1] - sc
    if abs(dr0) != abs(dc0) or abs(dr0) not in (1, 2):
        raise ValueError("not diagonal")

    if abs(dr0) == 1:
        if len(coords) != 2:
            raise ValueError("simple move must be exactly two squares")
        if _any_capture_from_board(b, player):
            raise ValueError("capture is mandatory")
        nr, nc = coords[1]
        if int(b[nr, nc]) != EMPTY:
            raise ValueError("destination occupied")
        piece = int(b[sr, sc])
        b[sr, sc] = EMPTY
        b[nr, nc] = piece
        _maybe_promote(b, nr, nc)
        return b

    cur_r, cur_c = sr, sc
    for i in range(1, len(coords)):
        nr, nc = coords[i]
        dr = nr - cur_r
        dc = nc - cur_c
        if abs(dr) != abs(dc) or abs(dr) != 2:
            raise ValueError("each jump must be distance 2")
        _apply_one_jump(b, cur_r, cur_c, nr, nc, player)
        cur_r, cur_c = nr, nc

    if _single_capture_landings(b, cur_r, cur_c, player):
        raise ValueError("multi-jump must continue with same piece")

    return b


def is_terminal(state: np.ndarray, player: int) -> Tuple[bool, Optional[int]]:
    """
    Terminal if someone won or current player cannot move (they lose).
    Draw (True, None) is not inferred from board alone (no move history); not used here.
    """
    b = np.asarray(state, dtype=np.int8)
    n_black = int(np.sum((b == BLACK_MAN) | (b == BLACK_KING)))
    n_red = int(np.sum((b == RED_MAN) | (b == RED_KING)))
    if n_black == 0:
        return (True, 2)
    if n_red == 0:
        return (True, 1)
    legal = get_legal_moves(b, player)
    if not legal:
        return (True, 2 if player == 1 else 1)
    return (False, None)


def board_to_ascii(state: np.ndarray) -> str:
    b = np.asarray(state, dtype=np.int8)
    sym = {
        EMPTY: ".",
        BLACK_MAN: "b",
        RED_MAN: "r",
        BLACK_KING: "B",
        RED_KING: "R",
    }
    lines: List[str] = []
    lines.append("   a b c d e f g h")
    for r in range(8):
        rank = 8 - r
        row_chars = [sym[int(b[r, c])] for c in range(8)]
        lines.append(f"{rank}  " + " ".join(row_chars) + f"  {rank}")
    lines.append("   a b c d e f g h")
    return "\n".join(lines)

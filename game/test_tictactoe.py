"""Tests for tic-tac-toe engine, matching test_checkers.py conventions."""

import numpy as np
import pytest

from game.tictactoe import (
    EMPTY,
    O_MARK,
    X_MARK,
    apply_move,
    board_to_ascii,
    format_square,
    get_legal_moves,
    initial_state,
    is_terminal,
    parse_square,
)
from game.validator import validate


def test_initial_state_empty():
    s = initial_state()
    assert s.shape == (3, 3)
    assert np.sum(s == EMPTY) == 9
    assert np.sum(s == X_MARK) == 0
    assert np.sum(s == O_MARK) == 0


def test_initial_legal_moves_nine_for_both():
    s = initial_state()
    x_moves = get_legal_moves(s, 1)
    o_moves = get_legal_moves(s, 2)
    assert len(x_moves) == 9
    assert x_moves == o_moves
    expected = sorted(
        [f"{c}{r}" for c in "abc" for r in "123"]
    )
    assert x_moves == expected


def test_apply_move_places_correct_mark():
    s = initial_state()
    s1 = apply_move(s, "a1", 1)  # bottom-left for X
    s2 = apply_move(s1, "b2", 2)  # center for O
    assert s1[2, 0] == X_MARK
    assert s2[1, 1] == O_MARK
    # original is unchanged
    assert np.all(s == 0)


def test_apply_move_rejects_occupied():
    s = apply_move(initial_state(), "b2", 1)
    with pytest.raises(ValueError):
        apply_move(s, "b2", 2)


def test_apply_move_rejects_bad_square():
    s = initial_state()
    with pytest.raises(ValueError):
        apply_move(s, "z9", 1)
    with pytest.raises(ValueError):
        apply_move(s, "a4", 1)
    with pytest.raises(ValueError):
        apply_move(s, "", 1)


def test_parse_format_roundtrip():
    for r in range(3):
        for c in range(3):
            sq = format_square(r, c)
            assert parse_square(sq) == (r, c)


def test_legal_moves_shrink_after_play():
    s = initial_state()
    s = apply_move(s, "a1", 1)
    assert len(get_legal_moves(s, 2)) == 8
    s = apply_move(s, "b2", 2)
    assert len(get_legal_moves(s, 1)) == 7


def test_row_win_detected():
    s = initial_state()
    s = apply_move(s, "a1", 1)  # (2,0)
    s = apply_move(s, "a3", 2)
    s = apply_move(s, "b1", 1)  # (2,1)
    s = apply_move(s, "b3", 2)
    s = apply_move(s, "c1", 1)  # (2,2) -> bottom row for X
    term, winner = is_terminal(s, 2)
    assert term is True
    assert winner == 1


def test_column_win_detected():
    s = initial_state()
    s = apply_move(s, "a1", 1)  # (2,0)
    s = apply_move(s, "b1", 2)
    s = apply_move(s, "a2", 1)  # (1,0)
    s = apply_move(s, "b2", 2)
    s = apply_move(s, "a3", 1)  # (0,0) -> left column
    term, winner = is_terminal(s, 2)
    assert term is True
    assert winner == 1


def test_diagonal_win_detected():
    s = initial_state()
    s = apply_move(s, "a1", 1)  # (2,0)
    s = apply_move(s, "a2", 2)
    s = apply_move(s, "b2", 1)  # (1,1)
    s = apply_move(s, "a3", 2)
    s = apply_move(s, "c3", 1)  # (0,2) -> anti-diagonal
    term, winner = is_terminal(s, 2)
    assert term is True
    assert winner == 1


def test_other_diagonal_win_detected():
    s = initial_state()
    s = apply_move(s, "c1", 1)  # (2,2)
    s = apply_move(s, "a1", 2)
    s = apply_move(s, "b2", 1)  # (1,1)
    s = apply_move(s, "c3", 2)
    s = apply_move(s, "a3", 1)  # (0,0) -> main diagonal
    term, winner = is_terminal(s, 2)
    assert term is True
    assert winner == 1


def test_draw_full_board():
    # X O X
    # X O O
    # O X X   (no 3 in a row)
    s = initial_state()
    moves = [
        ("b3", 1), ("a3", 2),  # center-top X, top-left O
        ("c3", 1), ("b2", 2),  # top-right X, center O
        ("a2", 1), ("c2", 2),  # mid-left X, mid-right O
        ("b1", 1), ("a1", 2),  # bot-center X, bot-left O
        ("c1", 1),             # bot-right X
    ]
    for mv, p in moves:
        s = apply_move(s, mv, p)
    assert np.sum(s == EMPTY) == 0
    term, winner = is_terminal(s, 1)
    assert term is True
    assert winner is None


def test_nonterminal_initial():
    s = initial_state()
    term, winner = is_terminal(s, 1)
    assert term is False
    assert winner is None


def test_validator_accepts_legal_and_rejects_illegal():
    s = initial_state()
    assert validate("a1", s, 1, game="tictactoe") is True
    s = apply_move(s, "a1", 1)
    assert validate("a1", s, 2, game="tictactoe") is False  # occupied
    assert validate("z9", s, 2, game="tictactoe") is False  # malformed
    assert validate("", s, 2, game="tictactoe") is False
    assert validate("b2", s, 2, game="tictactoe") is True


def test_board_to_ascii_shape():
    s = initial_state()
    s = apply_move(s, "b2", 1)
    out = board_to_ascii(s)
    lines = out.splitlines()
    assert len(lines) == 5  # header, 3 rows, footer
    assert "X" in out
    assert "a b c" in lines[0]

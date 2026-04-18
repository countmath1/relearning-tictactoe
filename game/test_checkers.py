"""Tests for American checkers engine and move validator."""

import numpy as np
import pytest

from game.checkers import (
    BLACK_KING,
    BLACK_MAN,
    RED_MAN,
    apply_move,
    board_to_ascii,
    get_legal_moves,
    initial_state,
    is_terminal,
)
from game.validator import validate


def test_initial_state_starting_position():
    s = initial_state()
    assert s.shape == (8, 8)
    assert np.sum(s == BLACK_MAN) == 12
    assert np.sum(s == RED_MAN) == 12
    assert np.sum(s == 0) == 64 - 24
    # Dark squares only in starting rows
    for r in range(3):
        for c in range(8):
            if (r + c) % 2 == 1:
                assert s[r, c] == BLACK_MAN
            else:
                assert s[r, c] == 0
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                assert s[r, c] == RED_MAN
            else:
                assert s[r, c] == 0


def test_opening_legal_moves_black_and_red():
    s = initial_state()
    black = get_legal_moves(s, 1)
    # Only rank-6 black men (row 2) can move: rank-7/8 forward squares are blocked by own men.
    expected_black = {
        "b6-a5",
        "b6-c5",
        "d6-c5",
        "d6-e5",
        "f6-e5",
        "f6-g5",
        "h6-g5",
    }
    assert set(black) == expected_black

    red = get_legal_moves(s, 2)
    # Only rank-3 red men (row 5) move; deeper ranks are blocked by own pieces.
    expected_red = {
        "a3-b4",
        "c3-b4",
        "c3-d4",
        "e3-d4",
        "e3-f4",
        "g3-f4",
        "g3-h4",
    }
    assert set(red) == expected_red


def test_mandatory_capture_filters_simple_moves():
    # Black man forward-captures only: d6 (2,3) jumps over c5 (3,2) to b4 (4,1).
    s = np.zeros((8, 8), dtype=np.int8)
    s[2, 3] = BLACK_MAN  # d6
    s[3, 2] = RED_MAN  # c5
    s[0, 7] = RED_MAN  # h8 — filler so red is not wiped out for terminal checks
    legal = get_legal_moves(s, 1)
    assert all("-" in m for m in legal)
    assert "d6-b4" in legal
    # No non-capturing shifts when a capture exists
    assert "d6-e5" not in legal


def test_multi_jump_forced_continuation():
    # Double jump along one diagonal: b6 x c5 x c3 -> b2
    s = np.zeros((8, 8), dtype=np.int8)
    s[2, 1] = BLACK_MAN  # b6
    s[3, 2] = RED_MAN  # c5
    s[5, 2] = RED_MAN  # c3
    s[0, 7] = RED_MAN  # h8 — filler
    moves = get_legal_moves(s, 1)
    assert moves == ["b6-d4-b2"]
    assert moves[0].count("-") == 2


def test_king_promotion_black_man():
    s = np.zeros((8, 8), dtype=np.int8)
    s[6, 2] = BLACK_MAN  # c2 one step from promotion
    s[0, 0] = RED_MAN  # keep red on board
    ns = apply_move(s, "c2-d1", 1)
    assert ns[7, 3] == BLACK_KING


def test_is_terminal_black_wins_no_red():
    s = np.zeros((8, 8), dtype=np.int8)
    s[0, 1] = BLACK_MAN
    t, w = is_terminal(s, 1)
    assert t and w == 1


def test_is_terminal_red_wins_no_black():
    s = np.zeros((8, 8), dtype=np.int8)
    s[0, 1] = RED_MAN
    t, w = is_terminal(s, 2)
    assert t and w == 2


def test_is_terminal_no_moves_current_player_loses():
    s = np.zeros((8, 8), dtype=np.int8)
    s[0, 1] = BLACK_MAN
    s[7, 6] = RED_MAN
    # Red to move but blocked? Single pieces far apart — red can always move?
    s = np.zeros((8, 8), dtype=np.int8)
    s[0, 1] = BLACK_MAN  # b8
    s[1, 0] = RED_MAN  # a7 blocks forward for black? black's turn with no moves?
    # Black at b8 (0,1), red at a7 (1,0) — black forward would be c7 empty...
    # Trap black: black man surrounded
    s = np.zeros((8, 8), dtype=np.int8)
    s[7, 0] = BLACK_MAN  # a1 corner — red man at b2 blocks one diag; need no legal moves
    # a1 black: can only go to b2 if empty
    s[6, 1] = RED_MAN  # b2 occupies
    # No other black pieces — black has no move from a1
    legal_b = get_legal_moves(s, 1)
    assert legal_b == []
    term, winner = is_terminal(s, 1)
    assert term and winner == 2


def test_validate_legal_and_illegal():
    s = initial_state()
    m = get_legal_moves(s, 1)[0]
    assert validate(m, s, 1) is True
    assert validate("z9-a1", s, 1) is False
    assert validate("", s, 1) is False
    assert validate("  ", s, 1) is False

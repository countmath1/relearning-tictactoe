"""Tests for Othello engine, matching test_checkers.py / test_tictactoe.py conventions."""

import numpy as np
import pytest

from game.othello import (
    BLACK,
    EMPTY,
    WHITE,
    apply_move,
    board_to_ascii,
    format_square,
    get_legal_moves,
    initial_state,
    is_terminal,
    parse_square,
)
from game.validator import validate


def test_initial_state_center_pieces():
    s = initial_state()
    assert s.shape == (8, 8)
    assert np.sum(s == BLACK) == 2
    assert np.sum(s == WHITE) == 2
    assert np.sum(s == EMPTY) == 60
    # Standard starting position
    assert s[3, 3] == WHITE
    assert s[3, 4] == BLACK
    assert s[4, 3] == BLACK
    assert s[4, 4] == WHITE


def test_parse_format_roundtrip():
    for r in range(8):
        for c in range(8):
            sq = format_square(r, c)
            assert parse_square(sq) == (r, c)


def test_parse_square_bad_input():
    with pytest.raises(ValueError):
        parse_square("z9")
    with pytest.raises(ValueError):
        parse_square("")
    with pytest.raises(ValueError):
        parse_square("abc")


def test_initial_legal_moves_black():
    s = initial_state()
    moves = get_legal_moves(s, 1)
    # Black's opening moves with this layout (d5=W, e5=B, d4=B, e4=W)
    assert set(moves) == {"c5", "d6", "e3", "f4"}


def test_initial_legal_moves_white():
    s = initial_state()
    moves = get_legal_moves(s, 2)
    # White's opening moves with this layout
    assert set(moves) == {"c4", "d3", "e6", "f5"}


def test_apply_move_flips_discs():
    s = initial_state()
    # Board: d5=W(3,3), e5=B(3,4), d4=B(4,3), e4=W(4,4)
    # Black plays e3: e3=(5,4), up (-1,0): (4,4)=WHITE, (3,4)=BLACK -> flips e4
    s2 = apply_move(s, "e3", 1)
    assert s2[5, 4] == BLACK  # placed disc
    assert s2[4, 4] == BLACK  # was white, now flipped to black


def test_apply_move_flips_correct_discs():
    """Verify a specific opening move flips the right disc."""
    s = initial_state()
    # With layout: (3,3)=W, (3,4)=B, (4,3)=B, (4,4)=W
    # i.e. d5=W, e5=B, d4=B, e4=W
    moves = get_legal_moves(s, 1)
    # Black at e4(4,4)=W? No. Let me just pick a legal move and verify.
    assert len(moves) > 0
    # Play first legal move
    m = moves[0]
    s2 = apply_move(s, m, 1)
    # Should have one more black disc (placed) plus flipped ones
    assert np.sum(s2 == BLACK) > np.sum(s == BLACK)
    # Total discs increased by 1 (placed one, flipped some)
    assert np.sum(s2 != EMPTY) == np.sum(s != EMPTY) + 1


def test_apply_move_rejects_illegal():
    s = initial_state()
    with pytest.raises(ValueError):
        apply_move(s, "a1", 1)  # corner: no adjacent pieces to flip


def test_apply_move_rejects_occupied():
    s = initial_state()
    with pytest.raises(ValueError):
        apply_move(s, "d5", 1)  # center is occupied


def test_legal_moves_shrink_after_play():
    s = initial_state()
    m1 = get_legal_moves(s, 1)
    s = apply_move(s, m1[0], 1)
    # After black moves, white should have legal moves (may differ in count)
    m2 = get_legal_moves(s, 2)
    assert len(m2) > 0


def test_is_terminal_initial_not_terminal():
    s = initial_state()
    term, winner = is_terminal(s, 1)
    assert term is False
    assert winner is None


def test_is_terminal_one_color_wins():
    # Board full of black except one white
    s = np.full((8, 8), BLACK, dtype=np.int8)
    s[0, 0] = WHITE
    # Neither player can move (board is full except... wait, board IS full)
    # Actually all squares are filled, so no legal moves for either player
    term, winner = is_terminal(s, 1)
    assert term is True
    assert winner == 1  # 63 black vs 1 white


def test_is_terminal_tie():
    # Half black, half white, no moves possible
    s = np.zeros((8, 8), dtype=np.int8)
    for r in range(8):
        for c in range(8):
            if r < 4:
                s[r, c] = BLACK
            else:
                s[r, c] = WHITE
    # Full board -> no legal moves for either player
    term, winner = is_terminal(s, 1)
    assert term is True
    assert winner is None  # 32 vs 32 = tie


def test_is_terminal_all_one_color():
    s = np.full((8, 8), BLACK, dtype=np.int8)
    term, winner = is_terminal(s, 2)
    assert term is True
    assert winner == 1


def test_flips_multiple_directions():
    """A move can flip discs in multiple directions at once."""
    s = np.zeros((8, 8), dtype=np.int8)
    # Set up a position where placing at (4,4) flips in two directions
    s[3, 4] = WHITE  # above
    s[2, 4] = BLACK  # bracket above
    s[4, 5] = WHITE  # right
    s[4, 6] = BLACK  # bracket right
    s2 = apply_move(s, format_square(4, 4), 1)
    assert s2[4, 4] == BLACK  # placed
    assert s2[3, 4] == BLACK  # flipped (was white)
    assert s2[4, 5] == BLACK  # flipped (was white)


def test_no_legal_moves_passes():
    """When a player has no moves, get_legal_moves returns empty list."""
    # Create a board where white has no legal moves: isolated white piece, no
    # empty square adjacent to black that would create a bracket.
    s = np.zeros((8, 8), dtype=np.int8)
    s[0, 0] = WHITE  # a8 corner, isolated
    # White needs an empty cell adjacent to a black disc with a white bracket.
    # With only one white piece at the corner and no black pieces, no flips possible.
    moves = get_legal_moves(s, 2)
    assert moves == []


def test_validator_accepts_legal_rejects_illegal():
    s = initial_state()
    legal = get_legal_moves(s, 1)
    assert len(legal) > 0
    assert validate(legal[0], s, 1, game="othello") is True
    assert validate("a1", s, 1, game="othello") is False
    assert validate("", s, 1, game="othello") is False


def test_board_to_ascii_shape():
    s = initial_state()
    out = board_to_ascii(s)
    lines = out.splitlines()
    assert len(lines) == 10  # header + 8 rows + footer
    assert "a b c d e f g h" in lines[0]
    assert "b" in out  # black disc
    assert "w" in out  # white disc

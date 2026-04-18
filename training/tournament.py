"""One tournament round: all pairwise games, logging, RL updates, checkpoints."""

from __future__ import annotations

from collections import defaultdict
from types import ModuleType
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

from game import get_game
from game.validator import validate
from models.pool import MODEL_CONFIGS, _build_prompt
from tracking.logger import Logger
from tracking.winrate import TARGET_MATCHUPS, WinRateTracker
from training.rl_update import run_update


def _unordered_pair_matches_target(
    black_idx: int, red_idx: int, attacker: int, defender: int
) -> bool:
    return {black_idx, red_idx} == {attacker, defender}


def _should_skip_game(
    black_idx: int, red_idx: int, completed_matchups: Set[Tuple[int, int]]
) -> bool:
    """Skip games between a target pair once that matchup has reached 50%."""
    if black_idx == red_idx:
        return False
    for a, d in completed_matchups:
        if _unordered_pair_matches_target(black_idx, red_idx, a, d):
            return True
    return False


def _play_one_game(
    pool: Any,
    black_idx: int,
    red_idx: int,
    trajectories: DefaultDict[int, List[Dict[str, Any]]],
    winrate_tracker: WinRateTracker,
    stats: Dict[str, List[int]],
    game: ModuleType,
    game_name: str,
) -> Tuple[str, int, bool, Optional[int], Optional[str]]:
    """
    Returns (outcome_key, num_moves, forfeit, forfeit_model_idx, bad_move_str).
    outcome_key: black_win | red_win | draw
    """
    state = game.initial_state()
    current_player = 1
    num_moves = 0
    max_moves = 8000

    game_steps: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)

    outcome_key = "draw"
    forfeit = False
    forfeit_model_idx: Optional[int] = None
    bad_move_str: Optional[str] = None
    winner_side: Optional[int] = None

    while num_moves < max_moves:
        idx = black_idx if current_player == 1 else red_idx
        legal = game.get_legal_moves(state, current_player)
        if not legal:
            winner_side = 2 if current_player == 1 else 1
            outcome_key = "black_win" if winner_side == 1 else "red_win"
            break

        move, log_prob = pool.generate_move(idx, state, current_player, legal, game=game_name)
        state_text = _build_prompt(state, current_player, legal, game=game_name)
        step = {
            "state_text": state_text,
            "action": move if move else "",
            "log_prob": float(log_prob),
            "reward": 0.0,
        }

        valid = bool(move) and validate(move, state, current_player, game=game_name)
        if not valid:
            forfeit = True
            forfeit_model_idx = idx
            bad_move_str = move
            winner_side = 2 if current_player == 1 else 1
            outcome_key = "black_win" if winner_side == 1 else "red_win"
            if MODEL_CONFIGS[idx]["trainable"]:
                game_steps[idx].append(step)
            break

        if MODEL_CONFIGS[idx]["trainable"]:
            game_steps[idx].append(step)

        state = game.apply_move(state, move, current_player)
        num_moves += 1

        next_p = 2 if current_player == 1 else 1
        term, w_side = game.is_terminal(state, next_p)
        if term:
            winner_side = w_side
            outcome_key = (
                "draw" if w_side is None
                else ("black_win" if w_side == 1 else "red_win")
            )
            break
        current_player = next_p

    if num_moves >= max_moves and winner_side is None:
        outcome_key = "draw"
        winner_side = None

    def reward_for_model(model_idx: int) -> float:
        if black_idx == red_idx:
            return 0.0
        if winner_side is None:
            return 0.0
        if winner_side == 1:
            return 1.0 if model_idx == black_idx else -1.0
        return 1.0 if model_idx == red_idx else -1.0

    for mid, steps in game_steps.items():
        r = reward_for_model(mid)
        for s in steps:
            s["reward"] = r
            trajectories[mid].append(s)

    # Stats: W/L/D/F per model index
    if black_idx == red_idx:
        pass
    elif outcome_key == "draw":
        stats["draws"][black_idx] += 1
        stats["draws"][red_idx] += 1
        winrate_tracker.record_game(black_idx, red_idx, False, draw=True)
        winrate_tracker.record_game(red_idx, black_idx, False, draw=True)
    else:
        bw = outcome_key == "black_win"
        stats["wins"][black_idx if bw else red_idx] += 1
        stats["losses"][red_idx if bw else black_idx] += 1
        if forfeit and forfeit_model_idx is not None:
            stats["forfeits"][forfeit_model_idx] += 1
        winrate_tracker.record_game(black_idx, red_idx, attacker_won=bw)
        winrate_tracker.record_game(red_idx, black_idx, attacker_won=not bw)

    return outcome_key, num_moves, forfeit, forfeit_model_idx, bad_move_str


def run_round(
    pool: Any,
    round_num: int,
    winrate_tracker: WinRateTracker,
    logger: Logger,
    completed_matchups: Set[Tuple[int, int]],
    game_name: str = "checkers",
) -> Set[Tuple[int, int]]:
    """
    Run all 16 ordered pair games unless skipped by completed_matchups.
    RL update per trainable model, checkpoints, logging, round table.
    Returns target matchups that newly crossed 50% this round.
    """
    game = get_game(game_name)
    n_models = len(MODEL_CONFIGS)
    stats: Dict[str, List[int]] = {
        "wins": [0] * n_models,
        "losses": [0] * n_models,
        "draws": [0] * n_models,
        "forfeits": [0] * n_models,
    }
    trajectories: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)

    for black_idx in range(n_models):
        for red_idx in range(n_models):
            if _should_skip_game(black_idx, red_idx, completed_matchups):
                continue
            game_id = f"r{round_num}_b{black_idx}_r{red_idx}"
            oc, nm, ff, ff_idx, bad = _play_one_game(
                pool,
                black_idx,
                red_idx,
                trajectories,
                winrate_tracker,
                stats,
                game,
                game_name,
            )
            logger.log_game(
                round_num,
                game_id,
                black_idx,
                red_idx,
                oc,
                nm,
                forfeit=ff,
                forfeit_model_idx=ff_idx,
                bad_move_str=bad,
            )

    # RL updates for trainable models (real weights only — not dry_run)
    for i, cfg in enumerate(MODEL_CONFIGS):
        if not cfg["trainable"]:
            continue
        traj = trajectories.get(i, [])
        if not traj:
            continue
        if getattr(pool, "dry_run", False):
            continue
        model = pool.models[i]
        tok = pool.tokenizers[i]
        if model is None:
            continue
        device = next(model.parameters()).device
        run_update(model, tok, traj, device)
        pool.save_checkpoint(i, round_num)

    stats["round"] = round_num
    logger.save_round(round_num, stats)
    logger.print_round_table(round_num, stats, winrate_tracker.get_summary())

    newly_done: Set[Tuple[int, int]] = set()
    for a, d in TARGET_MATCHUPS:
        if (a, d) in completed_matchups:
            continue
        if winrate_tracker.check_threshold(a, d):
            newly_done.add((a, d))
    return newly_done

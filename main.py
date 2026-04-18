"""RL tournament entry: pairwise checkers, rolling win rates, milestones, plots."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# Writable HF cache (cluster may set HF_HOME to an unwritable path).
_hf_home = os.environ.get("HF_HOME", "")
_default_hf = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
if not _hf_home:
    os.environ["HF_HOME"] = _default_hf
else:
    try:
        if not os.path.isdir(_hf_home):
            os.makedirs(_hf_home, exist_ok=True)
        elif not os.access(_hf_home, os.W_OK):
            os.environ["HF_HOME"] = _default_hf
    except OSError:
        os.environ["HF_HOME"] = _default_hf
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

from game import available_games  # noqa: E402
from models.pool import ModelPool  # noqa: E402
from tracking.logger import DISPLAY_NAMES, Logger  # noqa: E402
from tracking.plotter import save_winrate_plot  # noqa: E402
from tracking.winrate import WinRateTracker  # noqa: E402
from training.tournament import run_round  # noqa: E402


def _print_hardware() -> None:
    print("=== Hardware ===", flush=True)
    subprocess.run(["hostname"], check=False)
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            print(r.stdout.rstrip())
        else:
            print("(nvidia-smi failed; CPU-only is OK for --dry-run.)")
    except FileNotFoundError:
        print("(nvidia-smi not found; CPU-only is OK for --dry-run.)")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-model RL checkers tournament")
    p.add_argument("--rounds", type=int, default=500, help="Max rounds before stop")
    p.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling win-rate window size",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Random legal moves; no weights loaded",
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        help="4-bit quantization for loaded models",
    )
    p.add_argument(
        "--game",
        type=str,
        default="checkers",
        help="Game id (checkers or tictactoe)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.game.lower() not in available_games():
        print(
            f"Unsupported --game {args.game!r}; available: {available_games()}",
            file=sys.stderr,
        )
        sys.exit(1)

    _print_hardware()

    pool = ModelPool(dry_run=args.dry_run, quantize=args.quantize)
    winrate_tracker = WinRateTracker(DISPLAY_NAMES, window=args.window)
    logger = Logger()

    t0 = time.perf_counter()
    pool.load_all()
    print(f"[main] load_all() took {time.perf_counter() - t0:.1f}s\n")

    completed_matchups: set[tuple[int, int]] = set()
    milestone_rounds: dict[tuple[int, int], int] = {}
    milestone_games: dict[tuple[int, int], int] = {}
    total_games = 0

    names = [
        "Qwen2.5-0.5B",
        "Qwen2.5-1.5B",
        "Qwen2.5-3B",
        "Qwen2.5-7B",
    ]

    for round_num in range(1, args.rounds + 1):
        k_before = len(completed_matchups)
        newly_done = run_round(
            pool, round_num, winrate_tracker, logger, completed_matchups,
            game_name=args.game.lower(),
        )
        # Each completed target matchup removes two ordered games (both colors).
        total_games += 16 - 2 * k_before

        winrate_tracker.record_round_snapshot(round_num)
        save_winrate_plot(
            winrate_tracker.get_winrate_history(),
            names,
            path="results/winrate_curves.png",
        )

        for matchup in newly_done:
            milestone_rounds[matchup] = round_num
            milestone_games[matchup] = total_games
            completed_matchups.add(matchup)
            logger.log_milestone(
                matchup[0], matchup[1], round_num, total_games
            )

        if len(completed_matchups) == 3:
            print("\n=== ALL 3 TARGETS REACHED ===")
            break

    print("\n========== FINAL RESULTS ==========")
    for (a, d) in [(0, 1), (1, 2), (2, 3)]:
        if (a, d) not in milestone_rounds:
            continue
        rnd = milestone_rounds[(a, d)]
        games = milestone_games[(a, d)]
        print(
            f"{names[a]} reached 50% win rate vs {names[d]}: {rnd} rounds, {games} games"
        )
    if len(completed_matchups) < 3:
        print(
            f"Experiment ended at max rounds ({args.rounds}). Not all targets reached."
        )
        for (a, d) in [(0, 1), (1, 2), (2, 3)]:
            if (a, d) not in completed_matchups:
                wr = winrate_tracker.get_win_rate(a, d)
                print(
                    f"  {names[a]} vs {names[d]}: final win rate = {wr:.1%} (did not reach 50%)"
                )
    print("====================================")


if __name__ == "__main__":
    main()

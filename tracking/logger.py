"""Round and game logging for the RL tournament."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

# Display names aligned with models.pool.MODEL_CONFIGS order
DISPLAY_NAMES = [
    "Qwen2.5-0.5B",
    "Qwen2.5-1.5B",
    "Qwen2.5-3B",
    "Qwen2.5-7B",
]

WIN_RATE_LABELS = [
    "0.5B vs 1.5B:",
    "1.5B vs 3B:",
    "3B vs 7B:",
]


class Logger:
    def __init__(self, log_dir: str = "results/logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._games_path = self.log_dir / "games.log"

    def log_game(
        self,
        round_num: int,
        game_id: str,
        black_idx: int,
        red_idx: int,
        outcome: str,
        num_moves: int,
        forfeit: bool = False,
        forfeit_model_idx: int | None = None,
        bad_move_str: str | None = None,
    ) -> None:
        line = (
            f"round={round_num} game={game_id} black={black_idx} red={red_idx} "
            f"outcome={outcome} moves={num_moves} forfeit={forfeit}"
        )
        if forfeit_model_idx is not None:
            line += f" forfeit_model={forfeit_model_idx}"
        if bad_move_str is not None:
            line += f" bad_move={bad_move_str!r}"
        line += "\n"
        with open(self._games_path, "a", encoding="utf-8") as f:
            f.write(line)

    def log_milestone(
        self,
        attacker_idx: int,
        defender_idx: int,
        round_num: int,
        game_count: int,
    ) -> None:
        name = DISPLAY_NAMES[attacker_idx]
        opponent = DISPLAY_NAMES[defender_idx]
        msg = (
            f"Model {name} reached 50% win rate vs {opponent} after {round_num} rounds "
            f"({game_count} games)"
        )
        print(msg)
        path = self.log_dir / "milestones.log"
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def save_round(self, round_num: int, stats: dict) -> None:
        path = self.log_dir / f"round_{round_num}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    def print_round_table(
        self,
        round_num: int,
        stats: dict,
        winrates: Dict[Tuple[int, int], float],
    ) -> None:
        """Print ASCII table with per-model W/L/D/F and win% vs next larger model."""
        wins = stats.get("wins", [0] * 4)
        losses = stats.get("losses", [0] * 4)
        draws = stats.get("draws", [0] * 4)
        forfeits = stats.get("forfeits", [0] * 4)

        def win_rate_cell(row: int) -> str:
            if row == 3:
                return "—".ljust(29)
            pairs = [(0, 1), (1, 2), (2, 3)]
            label = WIN_RATE_LABELS[row]
            w = winrates.get(pairs[row], 0.0) * 100.0
            inner = f"{label}  {w:.1f}%"
            return inner.ljust(29)

        print(f"Round {round_num}")
        print(
            "┌─────────────────┬──────┬──────┬──────┬──────┬"
            "─────────────────────────────┐"
        )
        print(
            "│ Model           │  W   │  L   │  D   │  F   │"
            "  Win% vs next larger model  │"
        )
        print(
            "├─────────────────┼──────┼──────┼──────┼──────┼"
            "─────────────────────────────┤"
        )
        for i in range(4):
            dn = DISPLAY_NAMES[i]
            w, l, d, f = wins[i], losses[i], draws[i], forfeits[i]
            wr_pad = win_rate_cell(i)
            print(f"│ {dn:<15} │ {w:4d} │ {l:4d} │ {d:4d} │ {f:4d} │ {wr_pad} │")
        print(
            "└─────────────────┴──────┴──────┴──────┴──────┴"
            "─────────────────────────────┘"
        )

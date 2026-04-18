"""Rolling win-rate tracking for ordered model pairs (attacker vs defender)."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, DefaultDict, Deque, Dict, List, Tuple

# Smaller model attacks larger: 0.5B vs 1.5B, 1.5B vs 3B, 3B vs 7B
TARGET_MATCHUPS: List[Tuple[int, int]] = [(0, 1), (1, 2), (2, 3)]


class WinRateTracker:
    def __init__(self, model_names: List[str], window: int = 20) -> None:
        self.model_names = list(model_names)
        self.window = window
        # (attacker, defender) -> deque of outcome scores in [0, 0.5, 1] for the attacker
        self._history: DefaultDict[Tuple[int, int], Deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._total_games: DefaultDict[Tuple[int, int], int] = defaultdict(int)
        # Per-round rolling win rates for plotting (one entry per completed round)
        self._round_nums: List[int] = []
        self._rates_by_pair: Dict[Tuple[int, int], List[float]] = {
            pair: [] for pair in TARGET_MATCHUPS
        }

    def record_game(
        self,
        attacker_idx: int,
        defender_idx: int,
        attacker_won: bool,
        *,
        draw: bool = False,
    ) -> None:
        """
        Record one game from the attacker's perspective.

        - If draw is True, the attacker's outcome is 0.5 (and attacker_won is ignored).
        - Otherwise attacker_won=True means attacker won or defender forfeited;
          attacker_won=False means defender won or attacker forfeited.
        """
        key = (attacker_idx, defender_idx)
        self._total_games[key] += 1
        if draw:
            self._history[key].append(0.5)
        else:
            self._history[key].append(1.0 if attacker_won else 0.0)

    def get_win_rate(self, attacker_idx: int, defender_idx: int) -> float:
        dq = self._history[(attacker_idx, defender_idx)]
        if not dq:
            return 0.0
        return sum(dq) / len(dq)

    def check_threshold(
        self, attacker_idx: int, defender_idx: int, threshold: float = 0.5
    ) -> bool:
        dq = self._history[(attacker_idx, defender_idx)]
        if len(dq) < self.window:
            return False
        return (sum(dq) / len(dq)) >= threshold

    def get_summary(self) -> Dict[Tuple[int, int], float]:
        """Current rolling win rate for each target matchup."""
        return {pair: self.get_win_rate(pair[0], pair[1]) for pair in TARGET_MATCHUPS}

    def get_total_games(self, attacker_idx: int, defender_idx: int) -> int:
        """Total games recorded for this ordered pair (not capped by window)."""
        return self._total_games[(attacker_idx, defender_idx)]

    def record_round_snapshot(self, round_num: int) -> None:
        """Append current rolling win rates for each target matchup (call once per round)."""
        self._round_nums.append(round_num)
        for pair in TARGET_MATCHUPS:
            a, d = pair
            self._rates_by_pair[pair].append(self.get_win_rate(a, d))

    def get_winrate_history(self) -> Dict[str, Dict[str, Any]]:
        """
        History for plotting: keys are TARGET_MATCHUPS order labels
        '0.5B vs 1.5B', '1.5B vs 3B', '3B vs 7B' with 'rounds' and 'rates' lists.
        """
        labels = ["0.5B vs 1.5B", "1.5B vs 3B", "3B vs 7B"]
        out: Dict[str, Dict[str, Any]] = {}
        for i, pair in enumerate(TARGET_MATCHUPS):
            out[labels[i]] = {
                "rounds": list(self._round_nums),
                "rates": list(self._rates_by_pair[pair]),
            }
        return out

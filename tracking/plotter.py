"""Save rolling win-rate curves for the three target matchups."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MATCHUP_ORDER = ["0.5B vs 1.5B", "1.5B vs 3B", "3B vs 7B"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def save_winrate_plot(
    winrate_history: Dict[str, Dict[str, List[float]]],
    model_names: List[str],
    path: str = "results/winrate_curves.png",
) -> None:
    """
    Plot rolling win rate vs round for each target matchup.
    model_names is accepted for API compatibility; labels come from winrate_history keys.
    """
    _ = model_names
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="50%")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Round")
    ax.set_ylabel("Rolling win rate (attacker)")
    ax.set_title("Win rate vs next larger model (rolling window)")
    ax.grid(True, alpha=0.3)

    for i, key in enumerate(MATCHUP_ORDER):
        if key not in winrate_history:
            continue
        block = winrate_history[key]
        rounds = block.get("rounds", [])
        rates = block.get("rates", [])
        if not rounds or not rates or len(rounds) != len(rates):
            continue
        color = COLORS[i % len(COLORS)]
        ax.plot(rounds, rates, color=color, linewidth=2, label=key)

        # First round where rolling rate reaches or crosses 50%
        cross_idx = None
        for j, r in enumerate(rates):
            if r >= 0.5:
                cross_idx = j
                break
        if cross_idx is not None:
            rx = rounds[cross_idx]
            ry = rates[cross_idx]
            ax.scatter([rx], [ry], color=color, s=60, zorder=5, edgecolors="black", linewidths=0.5)
            ax.annotate(
                f" R{int(rx)}",
                (rx, ry),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color=color,
            )

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

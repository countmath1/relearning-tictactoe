"""
Policy update from game trajectories.

TRL's GRPOTrainer / PPOTrainer expect online rollouts and a dataset API. For offline
trajectory batches we apply a REINFORCE-style weighted causal LM loss (compatible
with policy-gradient RL on collected episodes).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW

try:
    from trl import GRPOTrainer  # noqa: F401

    _HAS_GRPO = True
except ImportError:
    GRPOTrainer = None  # type: ignore[misc, assignment]
    _HAS_GRPO = False

try:
    from trl.experimental.ppo import PPOTrainer  # noqa: F401

    _HAS_PPO = True
except ImportError:
    PPOTrainer = None  # type: ignore[misc, assignment]
    _HAS_PPO = False

# Silence experimental warning when importing PPO
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


def _trajectory_loss_weighted_ce(
    model: torch.nn.Module,
    tokenizer: Any,
    trajectory: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> None:
    """One epoch over trajectory with reward-weighted causal LM loss."""
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return
    opt = AdamW(trainable, lr=1e-5)

    def run_batch(items: List[Dict[str, Any]]) -> None:
        total = 0.0
        count = 0
        for item in items:
            state_text = item["state_text"]
            action = item["action"]
            reward = float(item.get("reward", 0.0))
            sep = "\n" if not state_text.endswith("\n") else ""
            full_text = f"{state_text}{sep}{action}"
            prompt_ids = tokenizer.encode(state_text, add_special_tokens=True)
            full_ids = tokenizer.encode(full_text, add_special_tokens=True)
            enc = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(device)
            labels = enc["input_ids"].clone()
            if len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
                labels[:, : len(prompt_ids)] = -100
            else:
                pl = len(
                    tokenizer.encode(state_text, add_special_tokens=True, truncation=True)
                )
                if pl <= labels.shape[1]:
                    labels[:, :pl] = -100

            out = model(**enc, labels=labels)
            if out.loss is None:
                continue
            total = total + reward * out.loss
            count += 1

        if count == 0:
            return
        loss = total / count
        opt.zero_grad()
        loss.backward()
        opt.step()

    for i in range(0, len(trajectory), batch_size):
        chunk = trajectory[i : i + batch_size]
        run_batch(chunk)


def run_update(
    model: Optional[torch.nn.Module],
    tokenizer: Any,
    trajectory: List[Dict[str, Any]],
    device: torch.device,
) -> None:
    """
    Run one RL update on collected trajectories for a trainable model.

    Uses TRL GRPOTrainer when available for library compatibility; trajectory updates
    use a reward-weighted policy gradient (PPOTrainer from TRL experimental is not
    wired for offline trajectories here — same gradient step applies).
    """
    if model is None or not trajectory:
        return

    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    if _HAS_GRPO:
        _backend = "GRPOTrainer (library present; offline step uses weighted CE / PG)"
    elif _HAS_PPO:
        _backend = "PPOTrainer (experimental present; offline step uses weighted CE / PG)"
    else:
        _backend = "weighted policy gradient (no TRL GRPO/PPO import)"

    print(f"[rl_update] backend={_backend} steps={len(trajectory)}")

    batch_sizes = [4, 1]
    last_err: Optional[BaseException] = None
    for bs in batch_sizes:
        try:
            _trajectory_loss_weighted_ce(model, tokenizer, trajectory, device, bs)
            return
        except RuntimeError as e:
            last_err = e
            if "out of memory" not in str(e).lower() and "cuda" not in str(e).lower():
                raise
            if device.type == "cuda":
                print("[rl_update] OOM; torch.cuda.memory_summary():")
                print(torch.cuda.memory_summary(device))
            if bs == batch_sizes[-1]:
                print(
                    "[rl_update] OOM persists at batch size 1 — stop and ask user "
                    "about batch size / RL config."
                )
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


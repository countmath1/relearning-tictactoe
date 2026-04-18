"""
Multi-model pool for Qwen2.5 Instruct checkpoints: LoRA on trainable models,
frozen eval model, dry-run and 4-bit options.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Default model list; assign devices via ModelPool(..., devices=[...]) or RL_MODEL_DEVICES.
MODEL_CONFIGS = [
    {"name": "Qwen/Qwen2.5-0.5B-Instruct", "trainable": True},
    {"name": "Qwen/Qwen2.5-1.5B-Instruct", "trainable": True},
    {"name": "Qwen/Qwen2.5-3B-Instruct", "trainable": True},
    {"name": "Qwen/Qwen2.5-7B-Instruct", "trainable": False},
]

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_ROOT = Path(
    os.environ.get("RL_TOURNAMENT_CHECKPOINT_DIR", str(ROOT / "checkpoints"))
)


def _parse_devices(devices: Optional[Sequence[Union[str, torch.device]]]) -> List[torch.device]:
    """Resolve per-model devices (length must match MODEL_CONFIGS)."""
    n = len(MODEL_CONFIGS)
    if devices is not None:
        if len(devices) != n:
            raise ValueError(f"devices must have length {n}, got {len(devices)}")
        return [torch.device(d) for d in devices]

    env = os.environ.get("RL_MODEL_DEVICES", "").strip()
    if env:
        parts = [p.strip() for p in env.split(",") if p.strip()]
        if len(parts) != n:
            raise ValueError(
                f"RL_MODEL_DEVICES must have {n} comma-separated entries, got {len(parts)}"
            )
        return [torch.device(p) for p in parts]

    # No explicit map: single GPU -> all on cuda:0; multi-GPU -> round-robin if enough devices.
    cnt = torch.cuda.device_count()
    if cnt == 0:
        print(
            "[ModelPool] No CUDA devices; using CPU for all models (slow / may OOM). "
            "Set devices=[...] or RL_MODEL_DEVICES=..."
        )
        return [torch.device("cpu")] * n
    if cnt >= n:
        return [torch.device(f"cuda:{i}") for i in range(n)]
    if cnt == 1:
        print(
            "[ModelPool] Single CUDA device: loading all models on cuda:0 (may OOM). "
            "Set devices=[...] or RL_MODEL_DEVICES=... to spread across GPUs."
        )
        return [torch.device("cuda:0")] * n
    # Fewer GPUs than models: round-robin
    print(
        f"[ModelPool] Round-robin assignment across cuda:0..cuda:{cnt - 1} "
        f"({n} models on {cnt} GPUs)."
    )
    return [torch.device(f"cuda:{i % cnt}") for i in range(n)]


def _print_cuda_mem(prefix: str, device: torch.device) -> None:
    if device.type != "cuda":
        return
    idx = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(idx)
    alloc = torch.cuda.memory_allocated(idx) / 1024**3
    reserved = torch.cuda.memory_reserved(idx) / 1024**3
    print(f"  [{prefix}] cuda:{idx} alloc={alloc:.2f} GiB reserved={reserved:.2f} GiB")


def _player_color(player: int, game: str = "checkers") -> str:
    if game == "checkers":
        return "black" if player == 1 else "red"
    if game == "othello":
        return "black" if player == 1 else "white"
    if game == "tictactoe":
        return "X" if player == 1 else "O"
    return "player 1" if player == 1 else "player 2"


def _build_prompt(
    state: np.ndarray,
    player: int,
    legal_moves: List[str],
    game: str = "checkers",
) -> str:
    if game == "checkers":
        from game.checkers import board_to_ascii
        game_name = "checkers"
        example = "example: e3-f4"
    elif game == "othello":
        from game.othello import board_to_ascii
        game_name = "othello"
        example = "example: d3"
    elif game == "tictactoe":
        from game.tictactoe import board_to_ascii
        game_name = "tic-tac-toe"
        example = "example: b2"
    else:
        raise ValueError(f"unknown game: {game!r}")

    ascii_board = board_to_ascii(state)
    color = _player_color(player, game)
    move_list = ", ".join(legal_moves)
    return (
        f"You are playing {game_name} as {color}. It is your turn.\n\n"
        f"Board (row 0 = top, column a = left):\n{ascii_board}\n\n"
        f"Legal moves: {move_list}\n\n"
        "Reply with exactly one move from the list above. No explanation, no extra words — "
        f"just the move ({example})."
    )


def _first_legal_move_in_text(text: str, legal_moves: List[str]) -> Optional[str]:
    """Earliest occurrence of any legal move substring in model output."""
    best: Optional[str] = None
    best_pos = len(text) + 1
    lower = text.lower()
    for m in legal_moves:
        ml = m.lower()
        pos = lower.find(ml)
        if pos != -1 and pos < best_pos:
            best_pos = pos
            best = m
    return best


_MOVE_RE = re.compile(r"\b[a-h][1-9](?:-[a-h][1-9])*\b", re.IGNORECASE)


def _first_legal_move_relaxed(text: str, legal_moves: List[str]) -> Optional[str]:
    """Try exact legal set first; then match regex candidates against legal set."""
    hit = _first_legal_move_in_text(text, legal_moves)
    if hit is not None:
        return hit
    legal_set = {m.lower() for m in legal_moves}
    best: Optional[str] = None
    best_pos = len(text) + 1
    for m in _MOVE_RE.finditer(text):
        cand = m.group(0)
        if cand.lower() in legal_set:
            # map back to canonical casing from legal_moves
            for lm in legal_moves:
                if lm.lower() == cand.lower():
                    if m.start() < best_pos:
                        best_pos = m.start()
                        best = lm
                    break
    return best


class ModelPool:
    def __init__(
        self,
        dry_run: bool = False,
        quantize: bool = False,
        devices: Optional[Sequence[Union[str, torch.device]]] = None,
    ) -> None:
        self.dry_run = dry_run
        self.quantize = quantize
        self._devices = _parse_devices(devices)
        self.tokenizers: List[Optional[AutoTokenizer]] = [None] * len(MODEL_CONFIGS)
        self.models: List[Optional[torch.nn.Module]] = [None] * len(MODEL_CONFIGS)
        self._lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def load_all(self) -> None:
        if self.dry_run:
            name = MODEL_CONFIGS[0]["name"]
            print(f"[ModelPool] dry_run: loading tokenizer only from {name}")
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            for i in range(len(MODEL_CONFIGS)):
                self.tokenizers[i] = tok
            print("[ModelPool] dry_run ready.")
            return

        bnb_config = None
        if self.quantize:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)

        for i, cfg in enumerate(MODEL_CONFIGS):
            name = cfg["name"]
            trainable = cfg["trainable"]
            device = self._devices[i]
            print(f"[ModelPool] Loading {i}: {name} on {device} (trainable={trainable})...")

            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self.tokenizers[i] = tok

            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                name,
                quantization_config=bnb_config,
                device_map={"": device} if device.type != "cpu" else None,
                torch_dtype=dtype if bnb_config is None else None,
                trust_remote_code=True,
            )
            if device.type == "cpu":
                model = model.to(device)

            if self.quantize and trainable:
                model = prepare_model_for_kbit_training(model)

            if trainable:
                model = get_peft_model(model, self._lora_config)
            else:
                model.requires_grad_(False)

            model.eval()
            self.models[i] = model
            _print_cuda_mem(f"after load {i}", device)

    def generate_move(
        self,
        model_idx: int,
        state: np.ndarray,
        player: int,
        legal_moves: List[str],
        game: str = "checkers",
    ) -> Tuple[Optional[str], float]:
        if not legal_moves:
            return (None, 0.0)

        if self.dry_run:
            return (random.choice(legal_moves), 0.0)

        model = self.models[model_idx]
        tokenizer = self.tokenizers[model_idx]
        if model is None or tokenizer is None:
            raise RuntimeError("Call load_all() before generate_move (non dry_run).")

        prompt = _build_prompt(state, player, legal_moves, game=game)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        device = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        seq = out.sequences[0]
        input_len = inputs["input_ids"].shape[1]
        new_tokens = seq[input_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

        move = _first_legal_move_relaxed(decoded, legal_moves)
        if move is None:
            return (None, 0.0)

        total_log_prob = 0.0
        if out.scores is not None and len(out.scores) > 0:
            for t, score in enumerate(out.scores):
                tid = new_tokens[t].item()
                lp = F.log_softmax(score[0], dim=-1)[tid]
                total_log_prob += float(lp.item())

        return (move, total_log_prob)

    def save_checkpoint(self, model_idx: int, round_num: int) -> None:
        cfg = MODEL_CONFIGS[model_idx]
        if not cfg["trainable"]:
            print(f"[ModelPool] model {model_idx} is frozen; skipping save.")
            return
        model = self.models[model_idx]
        if model is None:
            raise RuntimeError("Model not loaded.")
        out_dir = CHECKPOINT_ROOT / f"model_{model_idx}" / f"round_{round_num}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tok = self.tokenizers[model_idx]
        if tok is not None:
            tok.save_pretrained(out_dir)
        print(f"[ModelPool] Saved LoRA checkpoint to {out_dir}")

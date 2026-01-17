# src/budget.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time

@dataclass
class Usage:
    # “calls” is number of /api/generate requests
    calls: int = 0
    # token counts if Ollama returns them (prompt_eval_count / eval_count)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # wall time in seconds
    wall_s: float = 0.0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.wall_s += other.wall_s

@dataclass
class BudgetTracker:
    name: str
    usage: Usage = field(default_factory=Usage)
    curve: List[Dict[str, Any]] = field(default_factory=list)
    _t0: float = field(default_factory=time.time)

    def reset(self) -> None:
        self.usage = Usage()
        self.curve = []
        self._t0 = time.time()

    def log_point(self, tag: str, acc: float) -> None:
        self.curve.append({
            "tag": tag,
            "acc": float(acc),
            "calls": int(self.usage.calls),
            "prompt_tokens": int(self.usage.prompt_tokens),
            "completion_tokens": int(self.usage.completion_tokens),
            "wall_s": float(self.usage.wall_s),
            "since_start_s": float(time.time() - self._t0),
        })

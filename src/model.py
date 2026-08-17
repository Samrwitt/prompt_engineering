from __future__ import annotations

import hashlib
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

from src.tasks import solve_arithmetic, solve_boolean

LOCKED_MODEL_NAME = "llama3.2"


@dataclass
class LLMConfig:
    model_name: str = LOCKED_MODEL_NAME
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_predict: int = 512
    num_ctx: int = 1024
    timeout_s: int = 60


class LLMClient(Protocol):
    cfg: LLMConfig

    def generate(self, text: str, system: Optional[str] = None) -> str: ...

    def generate_with_usage(self, text: str, system: Optional[str] = None) -> Tuple[str, dict]: ...


def _empty_usage(wall: float = 0.0) -> dict:
    return {"calls": 1, "wall_s": wall, "prompt_tokens": 0, "completion_tokens": 0}


class OllamaLLM:
    def __init__(self, cfg: Optional[LLMConfig] = None) -> None:
        self.cfg = cfg or LLMConfig()
        self.cfg.model_name = LOCKED_MODEL_NAME
        try:
            import requests

            r = requests.get(f"{self.cfg.base_url}/api/tags", timeout=5)
            if r.status_code != 200:
                print(f"WARN: Ollama reachable but returned {r.status_code}")
        except Exception as e:
            print(f"CRITICAL: Could not connect to Ollama at {self.cfg.base_url}")
            print(f"Error: {e}")

    def generate_with_usage(self, text: str, system: Optional[str] = None) -> Tuple[str, dict]:
        import requests

        url = f"{self.cfg.base_url}/api/generate"
        payload = {
            "model": self.cfg.model_name,
            "prompt": text,
            "system": system or "",
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.num_predict,
                "num_ctx": self.cfg.num_ctx,
                "num_gpu": 99,
            },
        }

        t0 = time.time()
        res = requests.post(url, json=payload, timeout=self.cfg.timeout_s)
        wall = time.time() - t0

        if res.status_code != 200:
            return "", {"calls": 1, "wall_s": wall, "prompt_tokens": 0, "completion_tokens": 0}

        data = res.json()
        usage = {
            "calls": 1,
            "wall_s": wall,
            "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
            "completion_tokens": int(data.get("eval_count", 0) or 0),
        }
        return (data.get("response", "") or "").strip(), usage

    def generate(self, text: str, system: Optional[str] = None) -> str:
        out, _ = self.generate_with_usage(text, system=system)
        return out


def prompt_quality(system: str, user: str) -> float:
    """
    Heuristic quality of a prompt configuration.

    Higher scores mean the mock model is more likely to answer correctly
    and in a parseable format. This creates a real combinatorial landscape
    that metaheuristics can search without a live LLM.
    """
    s = f"{system}\n{user}".lower()
    score = 0.38
    if "precedence" in s or "not > and" in s:
        score += 0.20
    if "yes or no" in s or "exactly one token" in s:
        score += 0.14
    if "answer:" in s and "integer" in s:
        score += 0.14
    if "boolean" in s or "evaluate the given" in s or "mathematician" in s:
        score += 0.08
    if "step by step" in s or "reasoning" in s:
        score += 0.06
    n_shots = max(0, s.count("question:") - 1)
    score += min(0.16, 0.04 * n_shots)
    if not (system or "").strip():
        score -= 0.16
    return max(0.12, min(0.97, score))


def _extract_current_question(user: str) -> str:
    parts = re.split(r"Question:\s*", user)
    if len(parts) >= 2:
        last = parts[-1]
        last = re.split(r"\nAnswer", last, maxsplit=1)[0]
        return last.strip()
    return user.strip()


def _det_rng(*parts: str) -> random.Random:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


class MockLLM:
    """
    Deterministic stand-in for a local LLM.

    Used for tests, CI, and the interactive dashboard. Correctness probability
    depends on instruction/few-shot quality so the search problem stays real.
    """

    def __init__(
        self,
        cfg: Optional[LLMConfig] = None,
        oracle: Optional[Dict[str, str]] = None,
    ) -> None:
        self.cfg = cfg or LLMConfig()
        self.cfg.model_name = "mock"
        self.oracle = {str(k).strip().lower(): str(v) for k, v in (oracle or {}).items()}

    def _lookup(self, question: str) -> Optional[str]:
        key = question.strip().lower()
        if key in self.oracle:
            return self.oracle[key]
        for k, v in self.oracle.items():
            if k and k in key:
                return v
        return solve_boolean(question) or solve_arithmetic(question)

    def generate_with_usage(self, text: str, system: Optional[str] = None) -> Tuple[str, dict]:
        t0 = time.time()
        system = system or ""
        question = _extract_current_question(text)
        gold = self._lookup(question)
        quality = prompt_quality(system, text)
        rng = _det_rng(system, text)

        if gold is None:
            out = "I am not sure."
        elif rng.random() <= quality:
            if "exactly one token" in system.lower() or "yes or no" in system.lower():
                out = gold
            elif "answer:" in system.lower() and gold.lstrip("-").isdigit():
                out = f"Reasoning complete.\nAnswer: {gold}"
            else:
                out = f"The result is {gold}."
        else:
            if rng.random() < 0.5 and gold in ("yes", "no"):
                out = "no" if gold == "yes" else "yes"
            elif gold.lstrip("-").isdigit():
                noise = rng.choice([-3, -1, 1, 2, 5])
                out = str(int(gold) + noise)
            else:
                out = "The expression is complicated; maybe true?"

        wall = max(1e-4, time.time() - t0)
        usage = {
            "calls": 1,
            "wall_s": wall,
            "prompt_tokens": max(1, len(text) // 4),
            "completion_tokens": max(1, len(out) // 4),
        }
        return out, usage

    def generate(self, text: str, system: Optional[str] = None) -> str:
        out, _ = self.generate_with_usage(text, system=system)
        return out


def create_llm(
    backend: str = "mock",
    cfg: Optional[LLMConfig] = None,
    oracle: Optional[Dict[str, str]] = None,
) -> LLMClient:
    backend = (backend or "mock").lower()
    if backend in ("mock", "offline", "synthetic"):
        return MockLLM(cfg, oracle=oracle)
    if backend in ("ollama", "local"):
        return OllamaLLM(cfg)
    raise ValueError(f"Unknown LLM backend '{backend}'. Use 'mock' or 'ollama'.")

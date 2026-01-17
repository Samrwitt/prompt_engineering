from __future__ import annotations

import time
import requests
from dataclasses import dataclass
from typing import Optional, Tuple

LOCKED_MODEL_NAME = "llama3.2"

@dataclass
class LLMConfig:
    model_name: str = LOCKED_MODEL_NAME
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_predict: int = 512
    num_ctx: int = 1024
    timeout_s: int = 60

class OllamaLLM:
    def __init__(self, cfg: Optional[LLMConfig] = None) -> None:
        self.cfg = cfg or LLMConfig()
        self.cfg.model_name = LOCKED_MODEL_NAME

        # verify
        try:
            r = requests.get(f"{self.cfg.base_url}/api/tags", timeout=5)
            if r.status_code != 200:
                print(f"WARN: Ollama reachable but returned {r.status_code}")
        except Exception as e:
            print(f"CRITICAL: Could not connect to Ollama at {self.cfg.base_url}")
            print(f"Error: {e}")

    def generate_with_usage(self, text: str, system: Optional[str] = None) -> Tuple[str, dict]:
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
            return "", {"calls": 1, "wall_s": wall}

        data = res.json()
        # Ollama commonly returns prompt_eval_count and eval_count
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

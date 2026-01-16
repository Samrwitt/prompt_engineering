from __future__ import annotations

import requests
import json
from dataclasses import dataclass
from typing import Optional

# ==============================
# 🔒 GLOBAL MODEL LOCK
# ==============================
LOCKED_MODEL_NAME = "llama3.2"
  # or "tinyllama"
 


@dataclass
class LLMConfig:
    # Model is LOCKED — changing this has no effect
    model_name: str = LOCKED_MODEL_NAME

    # Ollama host
    base_url: str = "http://localhost:11434"
    
    # Deterministic decoding parameters
    temperature: float = 0.0
    num_predict: int = 64
    num_ctx: int = 512
    
    # Optional: context window size if needed, but Ollama defaults are usually fine
    
class OllamaLLM:
    """
    Research-safe LLM wrapper for Local Ollama.
    
    - Uses ONE fixed model (LOCKED_MODEL_NAME)
    - APIs into local Ollama instance
    - Deterministic decoding (temperature=0)
    """

    def __init__(self, cfg: Optional[LLMConfig] = None) -> None:
        self.cfg = cfg or LLMConfig()
        # 🔒 Force model name
        self.cfg.model_name = LOCKED_MODEL_NAME
        
        # Verify connection immediately
        try:
            res = requests.get(f"{self.cfg.base_url}/api/tags")
            if res.status_code != 200:
                print(f"WARN: Ollama reachable but returned {res.status_code}")
        except Exception as e:
            print(f"CRITICAL: Could not connect to Ollama at {self.cfg.base_url}. Is it running?")
            print(f"Error: {e}")

    def generate(self, text: str) -> str:
        """
        Generate answer text only.
        """
        url = f"{self.cfg.base_url}/api/generate"
        
        payload = {
            "model": self.cfg.model_name,
            "prompt": text,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": 64,  # Limit generation to avoid partial hangs
                # "seed": 42 
            }
        }
        
        try:
            res = requests.post(url, json=payload, timeout=30)  # Add 30s timeout
            if res.status_code == 200:
                data = res.json()
                return data.get("response", "").strip()
            else:
                print(f"Error generation: {res.text}")
                return ""
        except Exception as e:
            print(f"Exception during generation: {e}")
            return ""


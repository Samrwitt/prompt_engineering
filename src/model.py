from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass
class LLMConfig:
    model_name: str = "google/flan-t5-small"
    device: str = "cpu"
    max_new_tokens: int = 32


class HFLLM:
    def __init__(self, cfg: Optional[LLMConfig] = None) -> None:
        self.cfg = cfg or LLMConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name)
        self.model.to(self.cfg.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=False,  # deterministic
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

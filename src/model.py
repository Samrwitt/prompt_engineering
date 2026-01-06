from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)


@dataclass
class LLMConfig:
    model_name: str = "google/flan-t5-small"
    device: str = "cpu"  # "cuda" if you have GPU later
    max_new_tokens: int = 64

    # Keep deterministic for research reproducibility
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0


class HFLLM:
    """
    Unified wrapper for:
      - Seq2Seq models (e.g., FLAN-T5)
      - CausalLM models (e.g., Qwen/Phi/DeepSeek)

    You only change cfg.model_name.
    """

    def __init__(self, cfg: Optional[LLMConfig] = None) -> None:
        self.cfg = cfg or LLMConfig()

        # Load config first to detect architecture type
        self.hf_config = AutoConfig.from_pretrained(self.cfg.model_name, trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name,
            trust_remote_code=True,
            use_fast=True,
        )

        # Decide model class
        self.model_type: Literal["seq2seq", "causal"] = "seq2seq"
        if getattr(self.hf_config, "is_encoder_decoder", False):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.cfg.model_name,
                trust_remote_code=True,
            )
            self.model_type = "seq2seq"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_name,
                trust_remote_code=True,
            )
            self.model_type = "causal"

        self.model.to(self.cfg.device)
        self.model.eval()

        # Some CausalLM tokenizers don't have pad token set; fix for generate()
        if self.model_type == "causal":
            if self.tokenizer.pad_token_id is None:
                # fall back to eos as pad
                if self.tokenizer.eos_token_id is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    # very rare; create a pad token
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    self.model.resize_token_embeddings(len(self.tokenizer))

    @torch.inference_mode()
    def generate(self, text: str) -> str:
        """
        Returns ONLY the generated answer text (for CausalLM it strips the prompt).
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
        )

        # Deterministic generation: temperature/top_p only matter if do_sample=True
        if self.cfg.do_sample:
            gen_kwargs.update(dict(temperature=self.cfg.temperature, top_p=self.cfg.top_p))

        # Ensure pad token id is set for causal models
        if self.model_type == "causal":
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        out_ids = self.model.generate(**inputs, **gen_kwargs)

        if self.model_type == "seq2seq":
            # Seq2Seq output is only the generated tokens
            return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        # CausalLM output includes prompt + continuation; slice off prompt
        input_len = inputs["input_ids"].shape[1]
        cont_ids = out_ids[0][input_len:]
        return self.tokenizer.decode(cont_ids, skip_special_tokens=True).strip()

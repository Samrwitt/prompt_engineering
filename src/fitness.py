from __future__ import annotations

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from src.model import HFLLM

# Matches integers or decimals, including negative
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_number(text: str) -> str:
    """
    Extract a number from text (convenience wrapper for extract_answer).
    
    Args:
        text: Input text to extract number from
        
    Returns:
        Extracted number as string, or empty string if none found
    """
    return extract_answer(text, "number")


def load_dataset_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def split_train_test(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]


def load_blocks(path: str) -> List[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_prompt(blocks: List[str], x: List[int]) -> str:
    chosen = [b for b, bit in zip(blocks, x) if int(bit) == 1]
    return "\n".join(chosen).strip()


def extract_answer(text: str, answer_type: str) -> str:
    """
    answer_type: "number" | "yesno" | "abcd" | "text"
    Returns a normalized answer for scoring.
    """
    t = (text or "").strip().lower()

    if answer_type == "yesno":
        # Explicit yes/no
        if re.search(r"\byes\b", t):
            return "yes"
        if re.search(r"\bno\b", t):
            return "no"
        # Common variants
        if re.search(r"\btrue\b", t):
            return "yes"
        if re.search(r"\bfalse\b", t):
            return "no"
        # Numeric surrogates if model outputs 1/0
        if re.search(r"\b1\b", t):
            return "yes"
        if re.search(r"\b0\b", t):
            return "no"
        return ""

    if answer_type == "abcd":
        # Standalone A/B/C/D token only
        m = re.search(r"\b([abcd])\b", t)
        return m.group(1) if m else ""

    if answer_type == "number":
        # Prefer number after "answer" if present
        m = re.search(r"answer[^0-9-]*(-?\d+(?:\.\d+)?)", t)
        if m:
            return m.group(1)

        nums = _NUM_RE.findall(t.replace(",", ""))
        return nums[-1] if nums else ""

    # fallback (not recommended for strict scoring)
    return t


class PromptEvaluator:
    def __init__(
        self,
        llm: HFLLM,
        dataset: List[Dict[str, Any]],
        blocks: List[str],
        answer_type: str = "number",  # "yesno" | "abcd" | "number"
    ) -> None:
        self.llm = llm
        self.dataset = dataset
        self.blocks = blocks
        self.answer_type = answer_type

        # Cache: (prompt_vector, question_text) -> model_output
        self.cache: Dict[Tuple[Tuple[int, ...], str], str] = {}

    def _format_input(self, prompt_prefix: str, q: str) -> str:
        """
        Create a model input string. Keep this consistent for fair experiments.
        For causal LMs, trailing space after ":" helps completion; harmless for seq2seq.
        """
        if self.answer_type == "yesno":
            return (
                f"{prompt_prefix}\n\n"
                f"Question: {q}\n"
                f"Answer (yes or no): "
            )

        if self.answer_type == "abcd":
            return (
                f"{prompt_prefix}\n\n"
                f"Question: {q}\n"
                f"Answer (A/B/C/D): "
            )

        # default: number
        return (
            f"{prompt_prefix}\n\n"
            f"Question: {q}\n"
            f"Answer (integer only): "
        )

    def _normalize_ground_truth(self, a_raw: Any) -> str:
        a = str(a_raw).strip().lower()

        if self.answer_type == "yesno":
            if a in ("1", "yes", "true"):
                return "yes"
            if a in ("0", "no", "false"):
                return "no"
            return a

        if self.answer_type == "abcd":
            return a[:1]  # 'a'/'b'/'c'/'d'

        # number
        return a

    def eval_accuracy(self, x: List[int]) -> float:
        prompt_prefix = build_prompt(self.blocks, x)
        correct = 0
        total = len(self.dataset)

        for item in self.dataset:
            q = str(item["q"])
            gt = self._normalize_ground_truth(item["a"])

            key = (tuple(int(b) for b in x), q)
            if key in self.cache:
                out = self.cache[key]
            else:
                inp = self._format_input(prompt_prefix, q)
                out = self.llm.generate(inp)
                self.cache[key] = out

            pred = extract_answer(out, self.answer_type)

            if pred == gt:
                correct += 1

        return correct / total if total else 0.0

from __future__ import annotations

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

from src.model import HFLLM


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def load_dataset_jsonl(path: str) -> List[Dict[str, str]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def split_train_test(data: List[Dict[str, str]], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rng = random.Random(seed)
    # Shuffle a copy
    shuffled = list(data)
    rng.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]


def load_blocks(path: str) -> List[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_prompt(blocks: List[str], x: List[int]) -> str:
    chosen = [b for b, bit in zip(blocks, x) if int(bit) == 1]
    # You can tune this format later, but keep it fixed for fairness
    return "\n".join(chosen).strip()


def extract_number(text):
    t = text.lower()
    if "yes" in t: return "yes"
    if "no" in t: return "no"
    # New logic: map 1->yes, 0->no if explicit yes/no not found?
    # Or just treat them as synonyms.
    if "1" in t: return "yes"
    if "0" in t: return "no"
    if "a" in t: return "a"
    if "b" in t: return "b"
    return ""




class PromptEvaluator:
    def __init__(self, llm: HFLLM, dataset: List[Dict[str, str]], blocks: List[str]) -> None:
        self.llm = llm
        self.dataset = dataset
        self.blocks = blocks
        self.cache: Dict[Tuple[Tuple[int, ...], str], str] = {}

    def eval_accuracy(self, x: List[int]) -> float:
        prompt_prefix = build_prompt(self.blocks, x)
        correct = 0
        total = len(self.dataset)

        for item in self.dataset:
            q = item["q"]
            # Ensure ground truth is normalized if it's 1/0/yes/no
            a_raw = str(item["a"]).strip().lower()
            if a_raw == "1": a = "yes"
            elif a_raw == "0": a = "no"
            else: a = a_raw


            key = (tuple(x), q)
            if key in self.cache:
                out = self.cache[key]
            else:
                # FLAN-T5 pattern: Instruction + Input
                # We put the discovered blocks as the Instruction
                inp = (
    f"{prompt_prefix}\n"
    f"Question: {q}\n"
    f"Answer (yes or no):"
)

                out = self.llm.generate(inp)
                self.cache[key] = out

            pred = extract_number(out)
            if pred.strip() == a.strip():
                correct += 1

        return correct / total if total else 0.0

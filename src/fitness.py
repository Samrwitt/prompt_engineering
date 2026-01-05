from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

from src.model import HFLLM


_NUM_RE = re.compile(r"-?\d+(\.\d+)?")


def load_dataset_jsonl(path: str) -> List[Dict[str, str]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_blocks(path: str) -> List[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_prompt(blocks: List[str], x: List[int]) -> str:
    chosen = [b for b, bit in zip(blocks, x) if int(bit) == 1]
    # You can tune this format later, but keep it fixed for fairness
    return "\n".join(chosen).strip()



def extract_number(text: str) -> str:
    # Look for the last number in the string
    t = text.strip()
    nums = _NUM_RE.findall(t)
    if not nums:
        return ""
    # Return the last match (often the final answer)
    return nums[-1][0] if isinstance(nums[-1], tuple) else nums[-1]



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
            a = str(item["a"]).strip()


            key = (tuple(x), q)
            if key in self.cache:
                out = self.cache[key]
            else:
                # FLAN-T5 pattern: Instruction + Input
                # We put the discovered blocks as the Instruction
                inp = f"{prompt_prefix}\n\nQuestion: {q}\nAnswer:"
                out = self.llm.generate(inp)
                self.cache[key] = out

            pred = extract_number(out)
            if pred.strip() == a.strip():
                correct += 1

        return correct / total if total else 0.0

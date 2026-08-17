"""Prompt construction, answer extraction, and dataset I/O."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_YESNO_YES = re.compile(r"\b(yes|true)\b", re.I)
_YESNO_NO = re.compile(r"\b(no|false)\b", re.I)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "question" in obj and "q" not in obj:
                obj["q"] = obj["question"]
            if "answer" in obj and "a" not in obj:
                obj["a"] = obj["answer"]
            rows.append(obj)
    return rows


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_blocks(path: str | Path) -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"Instruction blocks file must be a JSON list of strings: {path}")
    return data


def split_train_test(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    n_train = int(len(data) * train_ratio)
    train = [data[i] for i in idx[:n_train]]
    test = [data[i] for i in idx[n_train:]]
    return train, test


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return int(math.ceil(len(text) / 4))


def build_prompt(blocks: List[str], x: List[int]) -> str:
    chosen = [b for b, bit in zip(blocks, x) if int(bit) == 1]
    return "\n".join(chosen).strip()


def extract_answer(text: str, answer_type: str) -> str:
    t = (text or "").strip().lower()

    if answer_type == "yesno":
        yes = _YESNO_YES.search(t)
        no = _YESNO_NO.search(t)
        if yes and not no:
            return "yes"
        if no and not yes:
            return "no"
        if yes and no:
            return "yes" if yes.start() < no.start() else "no"
        if re.search(r"\b1\b", t) and not re.search(r"\b0\b", t):
            return "yes"
        if re.search(r"\b0\b", t) and not re.search(r"\b1\b", t):
            return "no"
        return ""

    if answer_type == "abcd":
        m = re.search(r"\b([abcd])\b", t)
        return m.group(1) if m else ""

    if answer_type == "number":
        m = re.search(r"answer[^0-9-]*(-?\d+(?:\.\d+)?)", t)
        if m:
            return m.group(1)
        nums = _NUM_RE.findall(t.replace(",", ""))
        return nums[-1] if nums else ""

    return t


def normalize_gt(a_raw: Any, answer_type: str) -> str:
    a = str(a_raw).strip().lower()
    if answer_type == "yesno":
        if a in ("1", "yes", "true"):
            return "yes"
        if a in ("0", "no", "false"):
            return "no"
        return a
    if answer_type == "abcd":
        return a[:1]
    return a


def format_input(q: str, answer_type: str) -> str:
    if answer_type == "yesno":
        return f"Question: {q}\nAnswer (yes or no): "
    if answer_type == "abcd":
        return f"Question: {q}\nAnswer (A/B/C/D): "
    return f"Question: {q}\nAnswer (integer only): "


def decode_genome(
    x: List[int],
    n_blocks: int,
    n_demos: int,
    max_demos: int = 5,
) -> Tuple[List[int], List[int]]:
    """Split a joint genome into instruction bits and selected demo indices."""
    x_instr = list(x[:n_blocks])
    x_demo = list(x[n_blocks : n_blocks + n_demos])
    selected = [i for i, bit in enumerate(x_demo) if int(bit) == 1][:max_demos]
    return x_instr, selected

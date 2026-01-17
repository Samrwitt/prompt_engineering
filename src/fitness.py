from __future__ import annotations

import json, re, random, threading
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.model import OllamaLLM
from src.budget import BudgetTracker, Usage

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def load_dataset_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def split_train_test(data: List[Dict[str, Any]], train_ratio: float = 0.8, seed: int = 42):
    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]

def load_blocks(path: str) -> List[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def build_prompt(blocks: List[str], x: List[int]) -> str:
    chosen = [b for b, bit in zip(blocks, x) if int(bit) == 1]
    return "\n".join(chosen).strip()

def extract_answer(text: str, answer_type: str) -> str:
    t = (text or "").strip().lower()

    if answer_type == "yesno":
        if re.search(r"\byes\b", t): return "yes"
        if re.search(r"\bno\b", t): return "no"
        if re.search(r"\btrue\b", t): return "yes"
        if re.search(r"\bfalse\b", t): return "no"
        # avoid treating stray “1/0” inside text as label unless it’s alone-ish:
        if re.fullmatch(r"\s*1\s*", t): return "yes"
        if re.fullmatch(r"\s*0\s*", t): return "no"
        return ""

    if answer_type == "abcd":
        m = re.search(r"\b([abcd])\b", t)
        return m.group(1) if m else ""

    if answer_type == "number":
        m = re.search(r"answer[^0-9-]*(-?\d+(?:\.\d+)?)", t)
        if m: return m.group(1)
        nums = _NUM_RE.findall(t.replace(",", ""))
        return nums[-1] if nums else ""

    return t

class PromptEvaluator:
    def __init__(
        self,
        llm: OllamaLLM,
        dataset: List[Dict[str, Any]],
        blocks: List[str],
        answer_type: str = "number",
        tracker: Optional[BudgetTracker] = None,
        max_workers: int = 1,
    ) -> None:
        self.llm = llm
        self.dataset = dataset
        self.blocks = blocks
        self.answer_type = answer_type
        self.tracker = tracker
        self.max_workers = max_workers

        self.cache: Dict[Tuple[Tuple[int, ...], str], Tuple[str, dict]] = {}
        self._lock = threading.Lock()

    def _format_input(self, prompt_prefix: str, q: str) -> str:
        if self.answer_type == "yesno":
            return f"{prompt_prefix}\n\nQuestion: {q}\nAnswer (yes or no only): "
        if self.answer_type == "abcd":
            return f"{prompt_prefix}\n\nQuestion: {q}\nAnswer (A/B/C/D only): "
        return f"{prompt_prefix}\n\nQuestion: {q}\nAnswer (integer only): "

    def _normalize_ground_truth(self, a_raw: Any) -> str:
        a = str(a_raw).strip().lower()
        if self.answer_type == "yesno":
            if a in ("1", "yes", "true"): return "yes"
            if a in ("0", "no", "false"): return "no"
        if self.answer_type == "abcd":
            return a[:1]
        return a

    def _account_usage(self, usage_dict: dict) -> None:
        if not self.tracker:
            return
        u = Usage(
            calls=int(usage_dict.get("calls", 0) or 0),
            prompt_tokens=int(usage_dict.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage_dict.get("completion_tokens", 0) or 0),
            wall_s=float(usage_dict.get("wall_s", 0.0) or 0.0),
        )
        self.tracker.usage.add(u)

    def eval_accuracy(
        self,
        x: List[int],
        min_acc_to_beat: Optional[float] = None,   # early-stop threshold
        tag: Optional[str] = None,                 # curve label
    ) -> float:
        prompt_prefix = build_prompt(self.blocks, x)
        total = len(self.dataset)
        if total == 0:
            return 0.0

        xkey = tuple(int(b) for b in x)
        correct = 0
        seen = 0

        def one(item: Dict[str, Any]) -> int:
            q = str(item["q"])
            gt = self._normalize_ground_truth(item["a"])
            key = (xkey, q)

            with self._lock:
                cached = self.cache.get(key)

            if cached is not None:
                out, usage = cached
                # cache hit: do NOT account budget again
            else:
                inp = self._format_input(prompt_prefix, q)
                out, usage = self.llm.generate_with_usage(inp)
                with self._lock:
                    self.cache[key] = (out, usage)
                self._account_usage(usage)

            pred = extract_answer(out, self.answer_type)
            return 1 if pred == gt else 0

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures = [ex.submit(one, item) for item in self.dataset]
                for fut in as_completed(futures):
                    correct += int(fut.result())
                    seen += 1
                    if min_acc_to_beat is not None:
                        max_possible = (correct + (total - seen)) / total
                        if max_possible < min_acc_to_beat:
                            break
        else:
            for item in tqdm(self.dataset, desc="Eval", leave=False, disable=total < 5):
                correct += one(item)
                seen += 1
                if min_acc_to_beat is not None:
                    max_possible = (correct + (total - seen)) / total
                    if max_possible < min_acc_to_beat:
                        break

        acc = correct / max(1, seen) if seen < total else correct / total

        if self.tracker and tag:
            self.tracker.log_point(tag=tag, acc=acc)

        return float(acc)

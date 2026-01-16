# src/baseline.py
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Tuple, Optional

# Reuse your locked model + base_url from src.model
from src.model import LOCKED_MODEL_NAME, LLMConfig

# -----------------------------
# Legacy bit-vector baselines
# -----------------------------

def run_baseline(evaluator: Any) -> Tuple[List[int], float]:
    """Deterministic baseline: select no blocks (all zeros)."""
    n = len(evaluator.blocks)
    x = [0] * n
    acc = evaluator.eval_accuracy(x)
    return x, acc


def run_random_baseline(
    evaluator: Any,
    n_iter: int = 50,
    seed: int = 42
) -> Tuple[List[int], float]:
    """Random search baseline over 0/1 vectors."""
    rng = random.Random(seed)
    n = len(evaluator.blocks)
    best_x = [0] * n
    best_acc = -1.0

    for _ in range(n_iter):
        x = [rng.randint(0, 1) for _ in range(n)]
        acc = evaluator.eval_accuracy(x)
        if acc > best_acc:
            best_acc = acc
            best_x = x

    return best_x, best_acc


def run_greedy_baseline(
    evaluator: Any,
    max_iter: Optional[int] = None
) -> Tuple[List[int], float]:
    """Simple hill-climbing greedy baseline (bit-flip)."""
    n = len(evaluator.blocks)
    current_x = [0] * n
    current_acc = evaluator.eval_accuracy(current_x)

    if max_iter is None:
        max_iter = n * 2  # safety

    for _ in range(max_iter):
        best_neighbor_x = list(current_x)
        best_neighbor_acc = current_acc

        for i in range(n):
            neighbor = list(current_x)
            neighbor[i] = 1 - neighbor[i]
            acc = evaluator.eval_accuracy(neighbor)
            if acc > best_neighbor_acc:
                best_neighbor_acc = acc
                best_neighbor_x = neighbor

        if best_neighbor_acc > current_acc:
            current_x = best_neighbor_x
            current_acc = best_neighbor_acc
        else:
            break

    return current_x, current_acc


# -----------------------------
# DSPy (MIPROv2) baseline
# -----------------------------

def _norm_yesno(s: str) -> str:
    s = (s or "").strip().lower()
    if s.startswith(("y", "t")):
        return "yes"
    if s.startswith(("n", "f")):
        return "no"
    return s


def _extract_int(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r"-?\d+", s)
    return m.group(0) if m else ""


def _configure_dspy_for_ollama(base_url: str) -> None:
    """Configure DSPy to use local Ollama via LiteLLM."""
    import dspy
    lm = dspy.LM(
        f"ollama_chat/{LOCKED_MODEL_NAME}",
        api_base=base_url,
        api_key=""  # Ollama doesn't need a key
    )
    dspy.configure(lm=lm)


def _make_metric(answer_type: str):
    import dspy

    def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        gold = str(example.answer)
        got = str(pred.answer)

        if answer_type == "yesno":
            return 1.0 if _norm_yesno(gold) == _norm_yesno(got) else 0.0
        return 1.0 if _extract_int(gold) == _extract_int(got) else 0.0

    return metric


def _pick_key(row: Dict[str, Any], candidates: List[str], row_idx: int) -> Any:
    for k in candidates:
        if k in row:
            return row[k]
    raise KeyError(
        f"Row {row_idx} missing expected keys. Tried {candidates}. "
        f"Available keys: {sorted(list(row.keys()))}"
    )


def _to_dspy_examples(rows: List[Dict[str, Any]]):
    import dspy

    # ADD q/a here ✅
    q_keys = ["question", "q", "query", "input", "prompt", "problem", "text"]
    a_keys = ["answer", "a", "label", "output", "target", "gold"]

    examples: List[dspy.Example] = []
    for i, r in enumerate(rows):
        q = _pick_key(r, q_keys, i)
        a = _pick_key(r, a_keys, i)

        ex = dspy.Example({"question": str(q), "answer": str(a)}).with_inputs("question")
        examples.append(ex)

    return examples



def run_dspy_miprov2_baseline(
    train_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    answer_type: str,
    auto: str = "light",
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 4,
    seed: int = 0,
    base_url: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Real DSPy baseline using MIPROv2. Returns: (train_acc, test_acc).

    Requires:
      pip install dspy-ai litellm
    And Ollama running locally.
    """
    import dspy
    from dspy.teleprompt import MIPROv2

    cfg = LLMConfig()
    if base_url is None:
        base_url = cfg.base_url

    _configure_dspy_for_ollama(base_url)

    class NumberQA(dspy.Signature):
        question = dspy.InputField(desc="The math question")
        answer = dspy.OutputField(desc="The integer answer")

    class YesNoQA(dspy.Signature):
        question = dspy.InputField(desc="The logical question")
        answer = dspy.OutputField(desc="Yes or No")

    signature = YesNoQA if answer_type == "yesno" else NumberQA
    program = dspy.Predict(signature)

    metric = _make_metric(answer_type)

    trainset = _to_dspy_examples(train_rows)
    testset = _to_dspy_examples(test_rows)

    teleprompter = MIPROv2(metric=metric, auto=auto, seed=seed)

    optimized_program = teleprompter.compile(
        program.deepcopy(),
        trainset=trainset,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )

    def eval_acc(ds) -> float:
        scores = []
        for ex in ds:
            pred = optimized_program(question=ex.question)
            scores.append(metric(ex, pred))
        return float(sum(scores) / max(1, len(scores)))

    return eval_acc(trainset), eval_acc(testset)

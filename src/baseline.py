# src/baseline.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Optional

from src.model import LOCKED_MODEL_NAME, LLMConfig

# -----------------------------
# Strict parsing / normalization
# -----------------------------

_INT_RE = re.compile(r"^-?\d+$")

def _norm_yesno(s: str) -> str:
    s = (s or "").strip().lower()

    # accept common variants but normalize hard
    if s in ("yes", "y", "true", "t", "1"):
        return "yes"
    if s in ("no", "n", "false", "f", "0"):
        return "no"

    # if the model rambles, try to extract a clean yes/no token
    m = re.search(r"\b(yes|no|true|false)\b", s)
    if m:
        return "yes" if m.group(1) in ("yes", "true") else "no"
    return ""  # invalid

def _extract_int_strict(s: str) -> str:
    """
    Extract the final answer from text. 
    Looks for 'answer: X' or takes the last number in the text.
    """
    s = (s or "").strip().lower().replace(",", "")
    # Look for 'answer: X'
    m = re.search(r"answer[^0-9-]*(-?\d+)", s)
    if m:
        return m.group(1)
    # Fallback: take the last integer found
    nums = re.findall(r"-?\d+", s)
    return nums[-1] if nums else ""

def _is_valid_answer(answer_type: str, s: str) -> bool:
    if answer_type == "yesno":
        return _norm_yesno(s) in ("yes", "no")
    # number
    return _extract_int_strict(s) != ""

# -----------------------------
# DSPy + Ollama config
# -----------------------------

def _configure_dspy_for_ollama(base_url: str) -> None:
    import dspy
    lm = dspy.LM(
        f"ollama_chat/{LOCKED_MODEL_NAME}",
        api_base=base_url,
        api_key="",  # Ollama doesn't need a key
    )
    dspy.configure(lm=lm)

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

    q_keys = ["question", "q", "query", "input", "prompt", "problem", "text"]
    a_keys = ["answer", "a", "label", "output", "target", "gold"]

    examples: List[dspy.Example] = []
    for i, r in enumerate(rows):
        q = _pick_key(r, q_keys, i)
        a = _pick_key(r, a_keys, i)
        examples.append(dspy.Example(question=str(q), answer=str(a)).with_inputs("question"))
    return examples

# -----------------------------
# Metric (strict)
# -----------------------------

def _make_metric(answer_type: str):
    import dspy

    def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        gold = str(example.answer).strip()
        got = str(pred.answer).strip()

        if answer_type == "yesno":
            g = _norm_yesno(gold)
            p = _norm_yesno(got)
            return 1.0 if (g != "" and p != "" and g == p) else 0.0

        g = _extract_int_strict(gold)
        p = _extract_int_strict(got)
        return 1.0 if (g != "" and p != "" and g == p) else 0.0

    return metric

# -----------------------------
# Baseline: DSPy (MIPROv2)
# -----------------------------

def run_dspy_miprov2_baseline(
    train_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    answer_type: str,
    auto: str = "light",
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 4,
    seed: int = 0,
    base_url: Optional[str] = None,
    # constraints / robustness knobs
    max_pred_retries: int = 2,
) -> Tuple[float, float]:
    """
    DSPy baseline using MIPROv2. Returns: (train_acc, test_acc).

    Key idea: constrain outputs hard to avoid instruction drift and formatting noise.
    """
    import dspy
    from dspy.teleprompt import MIPROv2

    cfg = LLMConfig()
    if base_url is None:
        base_url = cfg.base_url
    _configure_dspy_for_ollama(base_url)

    # ---- SIGNATURES with strict output requirements ----
    class NumberQA(dspy.Signature):
        question = dspy.InputField(desc="Problem statement.")
        answer = dspy.OutputField(desc="Reason step-by-step, then output the final answer as exactly one integer (e.g., 42).")

    class YesNoQA(dspy.Signature):
        question = dspy.InputField(desc="Logical statement to evaluate.")
        answer = dspy.OutputField(desc="Reason step-by-step, then output exactly: yes or no.")

    signature = YesNoQA if answer_type == "yesno" else NumberQA
    program = dspy.Predict(signature)

    metric = _make_metric(answer_type)
    trainset = _to_dspy_examples(train_rows)
    testset = _to_dspy_examples(test_rows)

    teleprompter = MIPROv2(metric=metric, auto=auto, seed=seed)

    # Compile/optimize prompt parameters
    optimized_program = teleprompter.compile(
        program.deepcopy(),
        trainset=trainset,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )

    def _predict_with_retries(q: str) -> dspy.Prediction:
        """
        If the model violates the output contract, retry with a tighter reminder.
        This improves stability without changing the core algorithm.
        """
        reminder = ""
        for _ in range(max_pred_retries + 1):
            pred = optimized_program(question=(reminder + q))
            if _is_valid_answer(answer_type, str(pred.answer)):
                return pred

            # tighten reminder
            if answer_type == "yesno":
                reminder = (
                    "IMPORTANT: Output must be exactly one token: yes or no.\n"
                    "Do not explain.\n\n"
                )
            else:
                reminder = (
                    "IMPORTANT: Please provide your reasoning first, then conclude with the final integer.\n"
                    "The last number in your response should be the answer.\n\n"
                )
        return pred  # last attempt (may be invalid; metric will score 0)

    def eval_acc(ds) -> float:
        scores = []
        for ex in ds:
            pred = _predict_with_retries(ex.question)
            scores.append(metric(ex, pred))
        return float(sum(scores) / max(1, len(scores)))

    return eval_acc(trainset), eval_acc(testset)

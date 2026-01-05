from __future__ import annotations

from typing import List
from src.fitness import PromptEvaluator


def dspylike_baseline_vector(n_blocks: int) -> List[int]:
    """
    A strong fixed baseline:
    - include reasoning + constraints
    - exclude uncertain/guess blocks
    """
    # This assumes your blocks.json order above:
    # 0 careful, 1 expert, 2 step-by-step, 3 check, 4 final number, 5 no extra, 6 guess
    x = [0] * n_blocks
    for idx in [0, 1, 2, 3, 4, 5]:
        if idx < n_blocks:
            x[idx] = 1
    return x


def run_baseline(evaluator: PromptEvaluator) -> float:
    x0 = dspylike_baseline_vector(len(evaluator.blocks))
    return evaluator.eval_accuracy(x0)

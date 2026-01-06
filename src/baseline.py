from __future__ import annotations

import random
from typing import List, Tuple
from src.fitness import PromptEvaluator
from copy import deepcopy


def dspylike_baseline_vector(blocks: List[str]) -> List[int]:
    """
    DSPy-inspired fixed baseline prompt configuration.

    Idea:
    - Keep blocks that state the TASK and OUTPUT CONSTRAINTS
      (e.g., "solve", "answer directly", "single integer", "yes or no", "be precise").
    - Drop blocks that encourage guessing or unnecessary meta-talk
      (e.g., "if unsure, guess", "make your best guess").
    - Avoid chain-of-thought for small models ("step by step", "break down")
      unless you explicitly want it.

    This plays the role of a strong, deterministic "programmatic prompt"
    baseline similar in spirit to a DSPy program without teleprompt search.
    """

    x = [0] * len(blocks)

    for i, b in enumerate(blocks):
        text = b.lower()

        # 1) Always EXCLUDE guess / uncertainty language
        if "guess" in text or "unsure" in text or "best guess" in text:
            continue

        # 2) Optionally EXCLUDE step-by-step CoT for small models
        #    (you can flip this if you want CoT in baseline)
        if "step by step" in text or "break down" in text:
            continue

        # 3) INCLUDE task & constraint style instructions
        if (
            "solve" in text
            or "question" in text
            or "answer" in text
            or "be precise" in text
            or "precise" in text
            or "single integer" in text
            or "number only" in text
            or "do not include any explanation" in text
            or "no explanation" in text
            or "check your work" in text
            or "do not include explanation" in text
            or "answer directly" in text
            or "yes or no" in text
        ):
            x[i] = 1

    # 4) Fallback: if nothing selected, at least turn on the first block
    if not any(x) and len(blocks) > 0:
        x[0] = 1

    return x


def run_baseline(evaluator: PromptEvaluator) -> Tuple[List[int], float]:
    """
    Run the DSPy-like deterministic baseline:
    - Build a fixed "programmatic prompt" by including a subset of blocks.
    - No optimization, no randomness.
    """
    x0 = dspylike_baseline_vector(evaluator.blocks)
    return x0, evaluator.eval_accuracy(x0)


def run_random_baseline(evaluator: PromptEvaluator, num_samples: int = 20, seed: int = 42) -> Tuple[List[int], float]:
    """
    Random-search baseline:
    - Samples `num_samples` random 0/1 vectors.
    - Evaluates each.
    - Returns (best_vector, best_accuracy).
    """
    rng = random.Random(seed)
    n = len(evaluator.blocks)
    best_acc = -1.0
    best_x = [0] * n

    for _ in range(num_samples):
        # Sample random vector
        x = [rng.randint(0, 1) for _ in range(n)]
        # Ensure at least one block if all 0? Optional, but typical fitness handles empty.
        acc = evaluator.eval_accuracy(x)
        if acc > best_acc:
            best_acc = acc
            best_x = x
            
    return best_x, best_acc


def run_greedy_baseline(evaluator: PromptEvaluator, steps: int = 50, seed: int = 42) -> Tuple[List[int], float]:
    """
    Greedy hill-climbing baseline:
    - Starts from the DSPy-like baseline vector.
    - Repeatedly flips single bits if they improve accuracy.
    - Returns (best_vector, best_accuracy).
    """
    rng = random.Random(seed)
    n = len(evaluator.blocks)
    
    # Start: DSPy-like
    current_x = dspylike_baseline_vector(evaluator.blocks)
    current_acc = evaluator.eval_accuracy(current_x)
    
    best_x = list(current_x)
    best_acc = current_acc
    
    # Simple hill climbing: try flipping a random bit
    for _ in range(steps):
        # Pick a random bit to flip
        idx = rng.randint(0, n - 1)
        candidate_x = list(current_x)
        candidate_x[idx] = 1 - candidate_x[idx]
        
        acc = evaluator.eval_accuracy(candidate_x)
        
        # Accept if improves or equal (to drift) - usually strict improvement or explicit logic
        # Here we do greedy ascent: accept if >= current? or just >?
        # Let's say accept if >= to allow exploration on plateaus, but careful with loops.
        # User said "flips single bits if they improve accuracy", implying >.
        if acc > current_acc:
            current_acc = acc
            current_x = candidate_x
            if acc > best_acc:
                best_acc = acc
                best_x = list(candidate_x)
        # Else revert (don't update current_x)
    
    return best_x, best_acc

from __future__ import annotations

import random
from typing import List, Tuple
from src.fitness import PromptEvaluator
from src import dspy_local as dspy


# --- Define DSPy Signatures representing our Intent ---

class NumberQA(dspy.Signature):
    """
    Solve the math problem and answer with a single integer number.
    """
    question = dspy.InputField(desc="The math question")
    answer = dspy.OutputField(desc="The integer answer")

class YesNoQA(dspy.Signature):
    """
    Answer the question with Yes or No.
    """
    question = dspy.InputField(desc="The logical question")
    answer = dspy.OutputField(desc="Yes or No")


def dspylike_baseline_vector(blocks: List[str], answer_type: str = "number", use_cot: bool = False) -> List[int]:
    """
    Uses Local DSPy to 'compile' a baseline prompt from the available blocks.
    """
    # 1. Select Signature based on Answer Type
    if answer_type == "yesno":
        signature = YesNoQA
    else:
        signature = NumberQA
        
    # 2. Select Module Strategy (Predict vs CoT)
    if use_cot:
        module = dspy.ChainOfThought(signature)
    else:
        module = dspy.Predict(signature)
        
    # 3. Compile (find matching blocks)
    x = dspy.compile_to_vector(module, blocks)
    return x



def run_baseline(evaluator: PromptEvaluator) -> Tuple[List[int], float]:
    """
    Run the DSPy-like deterministic baseline.
    """
    # Use the evaluator's own answer_type to compile the prompt
    x0 = dspylike_baseline_vector(evaluator.blocks, answer_type=evaluator.answer_type)
    return x0, evaluator.eval_accuracy(x0)



def run_random_baseline(evaluator: PromptEvaluator, num_samples: int = 20, seed: int = 42) -> Tuple[List[int], float]:
    """
    Random-search baseline:
    - Samples `num_samples` random 0/1 vectors.
    """
    rng = random.Random(seed)
    n = len(evaluator.blocks)
    best_acc = -1.0
    best_x = [0] * n

    for _ in range(num_samples):
        # Sample random vector
        x = [rng.randint(0, 1) for _ in range(n)]
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
    """
    rng = random.Random(seed)
    n = len(evaluator.blocks)
    
    # Start: DSPy-like
    current_x = dspylike_baseline_vector(evaluator.blocks, answer_type=evaluator.answer_type)
    current_acc = evaluator.eval_accuracy(current_x)
    
    best_x = list(current_x)
    best_acc = current_acc
    
    
    for _ in range(steps):
        # Pick a random bit to flip
        idx = rng.randint(0, n - 1)
        candidate_x = list(current_x)
        candidate_x[idx] = 1 - candidate_x[idx]
        
        acc = evaluator.eval_accuracy(candidate_x)
        
        if acc > current_acc:
            current_acc = acc
            current_x = candidate_x
            if acc > best_acc:
                best_acc = acc
                best_x = list(candidate_x)
    
    return best_x, best_acc

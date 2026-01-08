"""
DSPy SOTA Baseline Integration

This module integrates DSPy's COPRO optimizer as a state-of-the-art baseline
for prompt optimization, allowing comparison with metaheuristic approaches.
"""

from __future__ import annotations

import json
from typing import List, Tuple, Optional

try:
    import dspy
    from dspy.teleprompt import COPRO
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None
    COPRO = None

from src.model import HFLLM
from src.fitness import load_dataset_jsonl, extract_answer


# --------------------------------------------------
# Local LM shim: make DSPy think it's talking to an LM,
# but internally we call your HFLLM().
# --------------------------------------------------
class LocalHFShim(dspy.LM):
    """Wrapper to make HFLLM compatible with DSPy's LM interface."""
    
    def __init__(self, llm: HFLLM):
        # "model" here is just a label; no real provider is used
        super().__init__(model="local/hfllm", model_type="text", cache=False)
        self.inner = llm

    def __call__(self, prompt: str | None = None, messages=None, **kwargs):
        # DSPy sometimes passes `messages` (chat style) and sometimes a plain `prompt`.
        if messages:
            text = ""
            # Prefer the last user message if available
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    text = m.get("content", "")
                    break
            if not text:
                last = messages[-1]
                if isinstance(last, dict):
                    text = last.get("content", "")
                else:
                    text = str(last)
        else:
            text = prompt or ""

        out = self.inner.generate(text)
        # DSPy LMs return a list of candidates; simplest is [string]
        return [out]


# --------------------------------------------------
# DSPy Signature and Program
# --------------------------------------------------
def create_qa_signature(answer_type: str = "number"):
    """Create a DSPy signature based on answer type."""
    if answer_type == "yesno":
        class QA(dspy.Signature):
            """Answer a yes/no question."""
            question: str = dspy.InputField()
            answer: str = dspy.OutputField(desc="yes or no only")
    elif answer_type == "number":
        class QA(dspy.Signature):
            """Answer a math question with a number."""
            question: str = dspy.InputField()
            answer: str = dspy.OutputField(desc="single integer only")
    else:
        class QA(dspy.Signature):
            """Answer a question with a short answer."""
            question: str = dspy.InputField()
            answer: str = dspy.OutputField(desc="short answer")
    
    return QA


class QAProgram(dspy.Module):
    """DSPy program for question answering."""
    
    def __init__(self, answer_type: str = "number"):
        super().__init__()
        QA = create_qa_signature(answer_type)
        self.predict = dspy.Predict(QA)
        self.answer_type = answer_type

    def forward(self, question: str):
        return self.predict(question=question)


# --------------------------------------------------
# Dataset loader (JSONL -> DSPy Examples)
# --------------------------------------------------
def load_dspy_dataset(path: str) -> List[dspy.Example]:
    """Load dataset from JSONL into DSPy Examples."""
    if not DSPY_AVAILABLE:
        raise ImportError("DSPy is not installed. Install with: pip install dspy-ai")
    
    dataset: List[dspy.Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            dataset.append(
                dspy.Example(
                    question=item["q"],
                    answer=item["a"],
                ).with_inputs("question")
            )
    return dataset


# --------------------------------------------------
# Metric: exact match with answer type normalization
# --------------------------------------------------
def create_exact_match_metric(answer_type: str = "number"):
    """Create an exact match metric function for the given answer type."""
    def exact_match(example, prediction, trace=None):
        pred_text = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
        pred_normalized = extract_answer(pred_text, answer_type)
        gt_normalized = extract_answer(str(example.answer), answer_type)
        return pred_normalized == gt_normalized
    
    return exact_match


# --------------------------------------------------
# Run DSPy COPRO (SOTA baseline)
# --------------------------------------------------
def run_dspy_sota(
    llm: HFLLM,
    train_data: List[dict],
    test_data: List[dict],
    answer_type: str = "number",
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Run DSPy COPRO optimizer as a SOTA baseline.
    
    Args:
        llm: HFLLM instance to use
        train_data: Training dataset (list of dicts with 'q' and 'a' keys)
        test_data: Test dataset
        answer_type: Type of answers ("number", "yesno", "abcd")
        verbose: Whether to print progress
        
    Returns:
        Tuple of (train_accuracy, test_accuracy)
    """
    if not DSPY_AVAILABLE:
        if verbose:
            print("Warning: DSPy not available. Skipping DSPy SOTA baseline.")
        return 0.0, 0.0
    
    try:
        # Configure DSPy with our LLM
        dspy.settings.configure(lm=LocalHFShim(llm))
        
        # Convert datasets to DSPy format
        train_dspy = [
            dspy.Example(question=item["q"], answer=item["a"]).with_inputs("question")
            for item in train_data
        ]
        test_dspy = [
            dspy.Example(question=item["q"], answer=item["a"]).with_inputs("question")
            for item in test_data
        ]
        
        # Create program
        program = QAProgram(answer_type=answer_type)
        
        # Create metric
        metric = create_exact_match_metric(answer_type)
        
        # Create optimizer
        optimizer = COPRO(
            metric=metric,
            max_bootstrapped_demos=min(4, len(train_dspy) // 2),
            max_labeled_demos=min(4, len(train_dspy) // 2),
            num_trials=10,
        )
        
        # Optimize on training set
        if verbose:
            print("  Optimizing with DSPy COPRO...")
        
        eval_kwargs = dict(
            num_threads=1,
            display_progress=False,
            display_table=0,
        )
        
        optimized_program = optimizer.compile(
            program,
            trainset=train_dspy,
            eval_kwargs=eval_kwargs,
        )
        
        # Evaluate on train and test
        train_correct = 0
        for ex in train_dspy:
            pred = optimized_program(question=ex.question)
            pred_text = pred.answer if hasattr(pred, 'answer') else str(pred)
            pred_normalized = extract_answer(pred_text, answer_type)
            gt_normalized = extract_answer(str(ex.answer), answer_type)
            if pred_normalized == gt_normalized:
                train_correct += 1
        
        test_correct = 0
        for ex in test_dspy:
            pred = optimized_program(question=ex.question)
            pred_text = pred.answer if hasattr(pred, 'answer') else str(pred)
            pred_normalized = extract_answer(pred_text, answer_type)
            gt_normalized = extract_answer(str(ex.answer), answer_type)
            if pred_normalized == gt_normalized:
                test_correct += 1
        
        train_acc = train_correct / len(train_dspy) if train_dspy else 0.0
        test_acc = test_correct / len(test_dspy) if test_dspy else 0.0
        
        return train_acc, test_acc
        
    except Exception as e:
        if verbose:
            print(f"  Error running DSPy SOTA: {e}")
        return 0.0, 0.0



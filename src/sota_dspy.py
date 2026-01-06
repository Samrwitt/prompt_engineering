import json
from typing import List

import dspy
from dspy.teleprompt import COPRO

# Reuse your local HF wrapper (google/flan-t5-small on CPU)
from src.model import HFLLM


# --------------------------------------------------
# Local LM shim: make DSPy think it's talking to an LM,
# but internally we call your HFLLM().
# --------------------------------------------------
class LocalHFShim(dspy.LM):
    def __init__(self):
        # "model" here is just a label; no real provider is used
        super().__init__(model="local/hfllm", model_type="text", cache=False)
        self.inner = HFLLM()

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


# Configure DSPy to use our local HF shim as the global LM
dspy.settings.configure(lm=LocalHFShim())


# --------------------------------------------------
# DSPy Signature and Program
# --------------------------------------------------
class QA(dspy.Signature):
    """Answer a question with a short answer."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="short answer")


class QAProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(QA)

    def forward(self, question: str):
        return self.predict(question=question)


# --------------------------------------------------
# Dataset loader (JSONL -> DSPy Examples)
# --------------------------------------------------
def load_dataset_jsonl(path: str) -> List[dspy.Example]:
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
# Metric: exact match
# --------------------------------------------------
def exact_match(example, prediction, trace=None):
    return (
        prediction.answer.strip().lower()
        == example.answer.strip().lower()
    )


# --------------------------------------------------
# Run DSPy COPRO (SOTA-ish baseline)
# --------------------------------------------------
def run_dspy_sota(dataset: list[dspy.Example]) -> float:
    program = QAProgram()

    optimizer = COPRO(
        metric=exact_match,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        num_trials=10,
    )

    # Newer DSPy requires eval_kwargs
    eval_kwargs = dict(
        num_threads=1,
        display_progress=False,
        display_table=0,
    )

    optimized_program = optimizer.compile(
        program,
        trainset=dataset,
        eval_kwargs=eval_kwargs,
    )

    # Evaluation on the same dataset (for now)
    correct = 0
    for ex in dataset:
        pred = optimized_program(question=ex.question)
        if pred.answer.strip().lower() == ex.answer.strip().lower():
            correct += 1

    return correct / len(dataset)



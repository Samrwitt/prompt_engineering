import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from src.model import OllamaLLM, LLMConfig
from src.experiment import BudgetedEvaluator, EvalConfig, BudgetStats, build_prompt
from typing import List, Optional, Tuple

class MockLLM(OllamaLLM):
    def __init__(self):
        self.last_system = None
        self.last_prompt = None
        self.cfg = LLMConfig() # dummy config

    def generate(self, text: str, system: Optional[str] = None) -> str:
        self.last_prompt = text
        self.last_system = system
        return "yes" # Dummy response

def verify_system_prompt_passing():
    print("Verifying System Prompt Passing & Few-Shot Demos...")
    mock_llm = MockLLM()
    blocks = ["Block A", "Block B"]
    
    # Create dummy candidates
    candidates = [
        {"q": "Q1", "a": "A1"},
        {"q": "Q2", "a": "A2"},
        {"q": "Q3", "a": "A3"},
        {"q": "Q4", "a": "A4"},
        {"q": "Q5", "a": "A5"},
        {"q": "Q6", "a": "A6"},
    ]
    
    budget = BudgetStats()
    cache = {}
    eval_cfg = EvalConfig(max_llm_calls=10, fast_k=5)
    
    evaluator = BudgetedEvaluator(
        llm=mock_llm,
        blocks=blocks,
        demo_candidates=candidates,
        answer_type="yesno",
        budget=budget,
        cache=cache,
        eval_cfg=eval_cfg
    )

    dataset = [{"q": "TargetQ", "a": "yes"}]

    # Test Case 1: All blocks + 2 demos
    # x = [1, 1] (blocks) + [1, 1, 0, 0, 0, 0] (demos)
    x = [1, 1, 1, 1, 0, 0, 0, 0]
    
    print("  Running evaluator.accuracy(x=blocks+demos)...")
    evaluator.accuracy(x, dataset)
    
    expected_system = "Block A\nBlock B"
    if mock_llm.last_system == expected_system:
        print("    [PASS] System prompt matches instructions.")
    else:
        print(f"    [FAIL] System prompt mismatch.\nExpected:\n{expected_system}\nGot:\n{mock_llm.last_system}")
    
    # Check if demos are in user prompt
    if "Question: Q1" in mock_llm.last_prompt and "Answer: A1" in mock_llm.last_prompt:
         print("    [PASS] Demo 1 present.")
    else:
         print("    [FAIL] Demo 1 missing.")

    if "Question: TargetQ" in mock_llm.last_prompt:
         print("    [PASS] Target question present.")
    else:
         print("    [FAIL] Target question missing.")

    # Test Case 2: Max demos limit (select all 6, should get 5)
    x_all_demos = [0, 0] + [1, 1, 1, 1, 1, 1]
    print("  Running evaluator.accuracy(x=all_demos)...")
    evaluator.accuracy(x_all_demos, dataset)
    
    if "Question: Q6" not in mock_llm.last_prompt:
        print("    [PASS] 6th demo excluded (limit 5).")
    else:
        print("    [FAIL] 6th demo present (limit failed).")

    # Test Case 3: No blocks, No demos
    x_none = [0, 0] + [0]*6
    print("  Running evaluator.accuracy(x=zeros)...")
    evaluator.accuracy(x_none, dataset)
    
    if mock_llm.last_system == "":
        print("    [PASS] System prompt is empty.")
    else:
        print(f"    [FAIL] System prompt not empty: '{mock_llm.last_system}'")

    if "Question: Q1" not in mock_llm.last_prompt:
        print("    [PASS] No demos present.")
    else:
        print("    [FAIL] Demos present when x_demo=0.")

if __name__ == "__main__":
    try:
        verify_system_prompt_passing()
        print("\nVerification Complete.")
    except Exception as e:
        print(f"\nVerification Failed with error: {e}")
        import traceback
        traceback.print_exc()

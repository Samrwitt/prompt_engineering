import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules["numpy"] = MagicMock()
sys.modules["tqdm"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Mock src.model to avoid importing it (which mocks torch/transformers effectively)
model_mock = MagicMock()
sys.modules["src.model"] = model_mock

# Now import the code to test
from src.fitness import extract_number, PromptEvaluator

def test_extract_number():
    print("Testing extract_number...")
    cases = [
        ("12", "12"),
        ("Answer: 42", "42"),
        ("The value is -3.14.", "-3.14"),
        ("Ignored 7 + 5 = 12", "12"),
        ("No number here", ""),
        ("Multiple: 1, 2, 3", "3"),
    ]
    for inp, expected in cases:
        got = extract_number(inp)
        status = "PASS" if got == expected else f"FAIL (got '{got}')"
        print(f"  Input: {inp!r} -> Expected: {expected!r} | {status}")

def test_evaluator_prompting():
    print("\nTesting PromptEvaluator prompting...")
    
    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "42"
    
    dataset = [{"q": "What is 6*7?", "a": "42"}]
    blocks = ["Think carefully."]
    
    evaluator = PromptEvaluator(mock_llm, dataset, blocks)
    
    # Run eval
    acc = evaluator.eval_accuracy([1])
    
    # Check what was sent to LLM
    expected_prompt = "Think carefully.\n\nCalculate: What is 6*7?\nAnswer:"
    call_args = mock_llm.generate.call_args
    if call_args:
        actual_prompt = call_args[0][0]
        if actual_prompt == expected_prompt:
            print("  Prompt Template: PASS")
        else:
            print(f"  Prompt Template: FAIL\n   Expected: {expected_prompt!r}\n   Actual:   {actual_prompt!r}")
    else:
        print("  Prompt Template: FAIL (LLM not called)")
        
    print(f"  Accuracy: {acc} (Expected 1.0)")

if __name__ == "__main__":
    test_extract_number()
    test_evaluator_prompting()

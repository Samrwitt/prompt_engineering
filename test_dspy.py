from src import dspy_local as dspy
from src.fitness import load_blocks
import json


def test_dspy_compilation():
    blocks_num = load_blocks("prompts/blocks_number.json")
    blocks_yn = load_blocks("prompts/blocks_yesno.json")
    
    class NumberQA(dspy.Signature):
        q = dspy.InputField(desc="Question")
        a = dspy.OutputField(desc="Answer")

    class YesNoQA(dspy.Signature):
        q = dspy.InputField(desc="Question")
        a = dspy.OutputField(desc="Answer")
        
    print("Test 1: Number Predict (Direct)...")
    mod = dspy.Predict(NumberQA)
    x = dspy.compile_to_vector(mod, blocks_num)
    chosen = [b for b, m in zip(blocks_num, x) if m]
    print("  Selected:", len(chosen))
    for c in chosen: print(f"    - {c}")
    
    assert any("Integer" in c or "number" in c.lower() for c in chosen), "Should pick integer constraints"
    assert not any("step by step" in c.lower() for c in chosen), "Should NOT pick CoT"
    
    print("\nTest 2: Number CoT...")
    mod_cot = dspy.ChainOfThought(NumberQA)
    x_cot = dspy.compile_to_vector(mod_cot, blocks_num)
    chosen_cot = [b for b, m in zip(blocks_num, x_cot) if m]
    print("  Selected:", len(chosen_cot))
    for c in chosen_cot: print(f"    - {c}")
        
    assert any("step by step" in c.lower() for c in chosen_cot), "Should pick CoT trigger"

    print("\nSUCCESS: DSPy logic verified against new blocks.")

if __name__ == "__main__":
    test_dspy_compilation()

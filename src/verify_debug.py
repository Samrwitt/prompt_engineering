from src.model import HFLLM
from src.fitness import extract_number, load_blocks, build_prompt

def main():
    blocks = load_blocks("prompts/blocks.json")
    # Turn all blocks on for the test
    x = [1] * len(blocks)
    prompt_prefix = build_prompt(blocks, x)
    
    # Test case "7 + 5" -> "12"
    q = "What is 7 + 5?"
    a = "12"
    
    # Use the same template as in fitness.py
    inp = f"{prompt_prefix}\n\nQuestion: {q}\nAnswer:"
    
    print("\n--- VERIFICATION DEBUG ---")
    print("INPUT:\n" + inp)
    print("--------------------------")
    
    llm = HFLLM()
    out = llm.generate(inp)
    print("OUTPUT:", repr(out))
    
    extracted = extract_number(out)
    print("EXTRACTED:", repr(extracted))
    
    if extracted == a:
        print("SUCCESS: Extracted number matches 12")
    else:
        print("FAILURE: Did not extract 12")

if __name__ == "__main__":
    main()

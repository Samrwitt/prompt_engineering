
import json
import random
from pathlib import Path

def save_jsonl(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def generate_toy_math(n=60):
    data = []
    # Mix of greater/less comparisons
    for _ in range(n):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        while a == b:
            b = random.randint(0, 100)
            
        if random.random() < 0.5:
            q = f"Is {a} greater than {b}?"
            ans = "yes" if a > b else "no"
        else:
            q = f"Is {a} less than {b}?"
            ans = "yes" if a < b else "no"
        data.append({"q": q, "a": ans})
    return data

def generate_arithmetic(n=60):
    data = []
    # 2-digit addition/subtraction
    for _ in range(n):
        op = random.choice(["+", "-"])
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        
        if op == "+":
            ans = a + b
        else:
            # Avoid negative for simplicity if desired, but let's allow it as it's arithmetic
            ans = a - b
            
        q = f"What is {a} {op} {b}?"
        data.append({"q": q, "a": str(ans)})
    return data

def generate_logic(n=60):
    data = []
    # Boolean logic: True/False AND/OR/NOT
    # "Is (True AND False) or True true?" -> yes/no
    # Let's keep it simple: "True AND False" -> "False" (or "no"?)
    # User said: "yes"/"no" answers.
    # So questions like: "Is (True AND False) true?" -> "no"
    
    vals = ["True", "False"] 
    ops = ["AND", "OR"]
    
    for _ in range(n):
        v1 = random.choice([True, False])
        v2 = random.choice([True, False])
        op = random.choice(ops)
        
        if op == "AND":
            res = v1 and v2
        else:
            res = v1 or v2
            
        # Maybe add a NOT
        if random.random() < 0.3:
            q_inner = f"not {str(v1)}"
            res = not v1
            q = f"Is {q_inner} true?"
        else:
            q = f"Is {v1} {op} {v2} true?"
            
        ans = "yes" if res else "no"
        data.append({"q": q, "a": ans})
        
    return data

if __name__ == "__main__":
    random.seed(42)
    
    toy = generate_toy_math(60)
    save_jsonl("data/toy_math.jsonl", toy)
    print(f"Saved {len(toy)} to data/toy_math.jsonl")
    
    arith = generate_arithmetic(60)
    save_jsonl("data/arithmetic.jsonl", arith)
    print(f"Saved {len(arith)} to data/arithmetic.jsonl")
    
    logic = generate_logic(60)
    save_jsonl("data/logic.jsonl", logic)
    print(f"Saved {len(logic)} to data/logic.jsonl")

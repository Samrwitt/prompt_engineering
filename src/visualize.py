import json
from pathlib import Path
import pandas as pd

def visualize_best_prompts():
    runs_path = Path("results/runs.jsonl")
    if not runs_path.exists():
        print("No run logs found.")
        return

    data = []
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    df = df.sort_values("test_acc", ascending=False)
    best_per_method = df.groupby(["dataset", "method"]).first().reset_index()
    
    # We might need to load training data to show demo text
    # But for now showing indices is a good start. 
    # Let's try to find the training files for each dataset
    ds_to_path = {
        "logic": "data/bbh_boolean_expressions.jsonl",
        "gsm8k": "data/gsm8k_sample.jsonl"
    }

    print("# Best Discovered Prompts by Method\n")
    
    for _, row in best_per_method.iterrows():
        ds_name = row['dataset']
        print(f"## Dataset: {ds_name} | Method: {row['method']}")
        print(f"- **Test Acc**: {row['test_acc']:.3f}")
        print(f"- **LLM Calls**: {row['budget']['llm_calls']}")
        
        print("\n### Selected Instructions:")
        print("```text")
        print(row['best_instruction_text'] or "[No Instructions]")
        print("```")
        
        demos = row.get('selected_demo_indices', [])
        if demos:
            print(f"\n### Selected Few-Shot Demos (Indices: {demos}):")
            path = ds_to_path.get(ds_name)
            if path and Path(path).exists():
                # Load training data (80% split)
                from src.experiment import load_jsonl, split_train_test
                full_data = load_jsonl(path)
                train_data, _ = split_train_test(full_data, seed=42)
                for idx in demos:
                    if idx < len(train_data):
                        d = train_data[idx]
                        print(f"- Q: {d['q'][:100]}... | A: {d['a']}")
            else:
                print(f"- Indices: {demos}")
        else:
            print("\n### Selected Few-Shot Demos: None")
            
        print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    visualize_best_prompts()

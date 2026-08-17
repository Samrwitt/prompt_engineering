import json
from pathlib import Path

import pandas as pd

from src.config import DATASET_CATALOG
from src.prompt import load_jsonl, split_train_test


def visualize_best_prompts(runs_path: str = "results/runs.jsonl") -> None:
    path = Path(runs_path)
    if not path.exists():
        print("No run logs found.")
        return

    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    df = df.sort_values("test_acc", ascending=False)
    best_per_method = df.groupby(["dataset", "method"], as_index=False).first()

    print("# Best Discovered Prompts by Method\n")
    for _, row in best_per_method.iterrows():
        ds_name = row["dataset"]
        budget = row["budget"] if isinstance(row["budget"], dict) else {}
        print(f"## Dataset: {ds_name} | Method: {row['method']}")
        print(f"- **Test Acc**: {row['test_acc']:.3f}")
        print(f"- **LLM Calls**: {budget.get('llm_calls', 'n/a')}")
        print("\n### Selected Instructions:")
        print("```text")
        print(row.get("best_instruction_text") or "[No Instructions]")
        print("```")

        demos = row.get("selected_demo_indices") or []
        if isinstance(demos, float):
            demos = []
        if demos:
            print(f"\n### Selected Few-Shot Demos (Indices: {list(demos)}):")
            spec = DATASET_CATALOG.get(ds_name)
            if spec and Path(spec["path"]).exists():
                full_data = load_jsonl(spec["path"])
                train_data, _ = split_train_test(full_data, seed=42)
                for idx in demos:
                    if idx < len(train_data):
                        d = train_data[idx]
                        q = str(d["q"])
                        snippet = q if len(q) < 120 else q[:117] + "..."
                        print(f"- Q: {snippet} | A: {d['a']}")
            else:
                print(f"- Indices: {list(demos)}")
        else:
            print("\n### Selected Few-Shot Demos: None")
        print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    visualize_best_prompts()

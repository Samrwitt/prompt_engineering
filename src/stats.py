import json
import numpy as np
from scipy.stats import wilcoxon
from collections import defaultdict
from pathlib import Path
import pandas as pd

def run_stats():
    runs_path = Path("results/runs.jsonl")
    if not runs_path.exists():
        print("No results/runs.jsonl found.")
        return

    data = []
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Group by Dataset
    datasets = df["dataset"].unique()
    
    results = {}
    
    for ds in datasets:
        print(f"\nAnalysis for Dataset: {ds}")
        sub = df[df["dataset"] == ds]
        
        # Pivot to get seeds as rows, methods as columns
        pivot = sub.pivot_table(index="seed", columns="method", values="test_acc")
        print(pivot.describe().T[["mean", "std", "count"]])
        
        methods = pivot.columns.tolist()
        # Assume Baseline is BASELINE_ALL or Random?
        baseline = "BASELINE_ALL" if "BASELINE_ALL" in methods else methods[0]
        
        print(f"\nSignificance vs {baseline} (Wilcoxon Signed-Rank):")
        sig_data = {}
        
        for m in methods:
            if m == baseline:
                continue
            
            # Get paired data (drop missing seeds)
            paired = pivot[[baseline, m]].dropna()
            
            if len(paired) < 5:
                res = "Ns (n<5)"
                p_val = 1.0
            else:
                try:
                    # Alternative: greater
                    w, p_val = wilcoxon(paired[m] - paired[baseline], alternative="greater")
                    res = f"p={p_val:.4f}"
                except ValueError:
                    # All differences zero
                    p_val = 1.0
                    res = "Identical"
            
            has_win = "*" if p_val < 0.05 else ""
            print(f"  {m:15s}: {res} {has_win}")
            sig_data[m] = {"p_value": p_val, "significant": p_val < 0.05}
            
        results[ds] = sig_data

    Path("results/significance.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_stats()

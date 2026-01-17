import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

def calculate_auc(curve, max_budget):
    """Calculates AUC for a step-function curve."""
    if not curve:
        return 0.0
    
    # Sort points by budget
    sorted_pts = sorted(curve, key=lambda x: x[0])
    
    # Add a point at (0, 0) if missing
    if sorted_pts[0][0] > 0:
        sorted_pts.insert(0, (0, 0))
    
    # Add a point at (max_budget, last_acc) to close the curve
    if sorted_pts[-1][0] < max_budget:
        sorted_pts.append((max_budget, sorted_pts[-1][1]))
    
    budgets = [p[0] for p in sorted_pts]
    accs = [p[1] for p in sorted_pts]
    
    # Use trapezoidal rule or step-wise integration
    # For convergence, step-wise is more accurate (constant acc until next improvement)
    auc = 0
    for i in range(len(sorted_pts) - 1):
        delta_x = sorted_pts[i+1][0] - sorted_pts[i][0]
        y = sorted_pts[i][1]
        auc += delta_x * y
        
    return auc / max_budget

def run_convergence_test():
    data = []
    # Merge both log files to capture all authentic data
    for filename in ["results/runss.jsonl", "results/runs.jsonl"]:
        p = Path(filename)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
    
    if not data:
        print("No result files found.")
        return
    
    df = pd.DataFrame(data)
    # Deduplicate keeping the latest entry for each (dataset, method, seed)
    df = df.drop_duplicates(subset=["dataset", "method", "seed"], keep="last")
    
    # Filter for methods with curves
    df = df[df["curve"].notnull()]
    
    results = {}
    
    for ds in df["dataset"].unique():
        print(f"\nConvergence Analysis (AUC) for Dataset: {ds}")
        sub = df[df["dataset"] == ds]
        
        # Max budget for this dataset
        max_budget = sub["budget"].apply(lambda x: x.get("llm_calls", 100)).max()
        
        # Calculate AUC for each run
        sub["auc"] = sub["curve"].apply(lambda c: calculate_auc(c, max_budget))
        
        # Pivot to get seeds as rows, methods as columns
        pivot = sub.pivot_table(index="seed", columns="method", values="auc")
        print("\nMean AUC per Method (Higher is faster/better convergence):")
        print(pivot.mean().sort_values(ascending=False))
        
        methods = pivot.columns.tolist()
        baseline = "BASELINE_ALL" if "BASELINE_ALL" in methods else ( "dspy_miprov2" if "dspy_miprov2" in methods else methods[0])
        
        print(f"\nSignificance (AUC) vs {baseline}:")
        sig_data = {}
        
        for m in methods:
            if m == baseline:
                continue
            
            paired = pivot[[baseline, m]].dropna()
            
            if len(paired) < 3: # Lower threshold for AUC since it's more stable
                res = "Ns (n<3)"
                p_val = 1.0
            else:
                try:
                    # Test if m AUC > baseline AUC
                    w, p_val = wilcoxon(paired[m] - paired[baseline], alternative="greater")
                    res = f"p={p_val:.4f}"
                except ValueError:
                    p_val = 1.0
                    res = "Identical"
            
            has_win = "*" if p_val < 0.05 else ""
            print(f"  {m:15s}: {res} {has_win}")
            sig_data[m] = {
                "mean_auc": float(pivot[m].mean()), 
                "p_value": float(p_val), 
                "significant": bool(p_val < 0.05)
            }
            
        results[ds] = sig_data

    Path("results/convergence_tests.json").write_text(json.dumps(results, indent=2))
    print(f"\nConvergence results saved to results/convergence_tests.json")

if __name__ == "__main__":
    run_convergence_test()

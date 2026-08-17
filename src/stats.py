from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import wilcoxon
import pandas as pd


def load_runs_frame(path: Path | None = None) -> pd.DataFrame:
    runs_path = path or Path("results/runs.jsonl")
    if not runs_path.exists():
        return pd.DataFrame()
    data: List[Dict[str, Any]] = []
    with runs_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)


def paired_wilcoxon(series_a: pd.Series, series_b: pd.Series) -> Dict[str, Any]:
    paired = pd.concat([series_a, series_b], axis=1).dropna()
    paired.columns = ["baseline", "method"]
    n = len(paired)
    if n < 5:
        return {"p_value": None, "significant": False, "n": n, "note": "n<5"}
    diff = paired["method"] - paired["baseline"]
    if np.allclose(diff, 0):
        return {"p_value": 1.0, "significant": False, "n": n, "note": "identical"}
    try:
        _w, p_val = wilcoxon(diff, alternative="greater")
    except ValueError:
        return {"p_value": 1.0, "significant": False, "n": n, "note": "wilcoxon failed"}
    return {"p_value": float(p_val), "significant": bool(p_val < 0.05), "n": n}


def run_stats(runs_path: str = "results/runs.jsonl") -> Dict[str, Any]:
    df = load_runs_frame(Path(runs_path))
    if df.empty:
        print(f"No results at {runs_path}.")
        return {}

    results: Dict[str, Any] = {}
    for ds in sorted(df["dataset"].dropna().unique()):
        print(f"\nAnalysis for Dataset: {ds}")
        sub = df[df["dataset"] == ds]
        pivot = sub.pivot_table(index="seed", columns="method", values="test_acc")
        print(pivot.describe().T[["mean", "std", "count"]])
        methods = list(pivot.columns)
        baseline = "BASELINE_ALL" if "BASELINE_ALL" in methods else methods[0]
        print(f"\nSignificance vs {baseline} (Wilcoxon signed-rank, alternative=greater):")
        sig_data: Dict[str, Any] = {}
        for m in methods:
            if m == baseline:
                continue
            stats = paired_wilcoxon(pivot[baseline], pivot[m])
            p_val = stats["p_value"]
            if p_val is None:
                res = f"n={stats['n']} (need ≥5 paired seeds)"
                star = ""
            else:
                res = f"p={p_val:.4f} n={stats['n']}"
                star = "*" if stats["significant"] else ""
            print(f"  {m:15s}: {res} {star}")
            sig_data[m] = stats
        results[ds] = {"baseline": baseline, "tests": sig_data}

    Path("results/significance.json").write_text(json.dumps(results, indent=2))
    print("\nWrote results/significance.json")
    return results


if __name__ == "__main__":
    run_stats()

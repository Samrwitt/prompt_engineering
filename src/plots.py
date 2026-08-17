"""Publication-quality figures from logged experiment runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

METHOD_ORDER = [
    "BASELINE_NONE",
    "BASELINE_ALL",
    "RANDOM",
    "GREEDY",
    "HILL_CLIMB",
    "SA+",
    "DE",
    "GWO",
    "HYBRID_DE_SA",
    "dspy_miprov2",
]

COLORS = {
    "BASELINE_ALL": "#4B5563",
    "BASELINE_NONE": "#9CA3AF",
    "RANDOM": "#D97706",
    "GREEDY": "#CA8A04",
    "HILL_CLIMB": "#A16207",
    "SA+": "#DC2626",
    "DE": "#2563EB",
    "GWO": "#059669",
    "HYBRID_DE_SA": "#7C3AED",
    "dspy_miprov2": "#0F766E",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_runs(results_dir: Path | None = None, include_legacy: bool = False) -> pd.DataFrame:
    root = _project_root()
    results_dir = results_dir or (root / "results")
    names = ["runs.jsonl"] + (["runss.jsonl"] if include_legacy else [])
    rows: List[Dict[str, Any]] = []
    for name in names:
        path = results_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["dataset", "method", "seed"], keep="last")
    return df


def load_final_scores(results_dir: Path | None = None) -> Dict[str, Any]:
    root = _project_root()
    path = (results_dir or root / "results") / "final_scores.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)


def _accuracy_summary(df: pd.DataFrame, dataset: str, scores: Dict[str, Any] | None = None) -> pd.DataFrame:
    if scores and dataset in scores:
        rows = []
        for method, payload in scores[dataset].items():
            if isinstance(payload, dict) and "mean_test" in payload:
                rows.append({
                    "method": method,
                    "mean": float(payload["mean_test"]),
                    "std": float(payload.get("std_test") or 0.0),
                    "count": int(payload.get("runs") and len(payload["runs"]) or payload.get("count") or 0),
                })
            elif isinstance(payload, dict) and "test" in payload:
                rows.append({
                    "method": method,
                    "mean": float(payload["test"]),
                    "std": 0.0,
                    "count": 1,
                })
        if rows:
            summary = pd.DataFrame(rows).set_index("method")
            return summary.reindex([m for m in METHOD_ORDER if m in summary.index]).dropna(how="all")
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby("method")["test_acc"]
        .agg(["mean", "std", "count"])
        .reindex([m for m in METHOD_ORDER if m in set(sub["method"])])
        .dropna(how="all")
    )


def plot_accuracy_bars(
    df: pd.DataFrame,
    out_path: Path,
    dataset: str = "logic",
    scores: Dict[str, Any] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _accuracy_summary(df, dataset, scores)
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(summary))
    colors = [COLORS.get(m, "#374151") for m in summary.index]
    mean = summary["mean"].to_numpy(dtype=float)
    std = summary["std"].fillna(0).to_numpy(dtype=float)
    lower = np.minimum(std, mean)
    upper = np.minimum(std, 1.0 - mean)
    ax.bar(x, mean, yerr=[lower, upper], capsize=4, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    _style_axes(ax, f"Mean test accuracy — {dataset}", "Method", "Test accuracy")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _step_curve(curve: List[List[float]], max_budget: int) -> np.ndarray:
    pts = sorted((int(a), float(b)) for a, b in curve if a is not None)
    y = np.zeros(max_budget + 1)
    val = 0.0
    idx = 0
    for x in range(max_budget + 1):
        while idx < len(pts) and pts[idx][0] <= x:
            val = pts[idx][1]
            idx += 1
        y[x] = val
    return y


def plot_convergence(df: pd.DataFrame, out_path: Path, dataset: str = "logic", max_budget: int = 100) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[df["dataset"] == dataset]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    plotted = 0
    for method in METHOD_ORDER:
        rows = sub[sub["method"] == method]
        series = []
        for _, row in rows.iterrows():
            curve = row.get("curve")
            if isinstance(curve, list) and curve:
                series.append(_step_curve(curve, max_budget))
        if not series:
            continue
        mat = np.vstack(series)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        xs = np.arange(max_budget + 1)
        color = COLORS.get(method, "#111827")
        ax.plot(xs, mean, label=method, color=color, linewidth=2)
        ax.fill_between(xs, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.12)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    _style_axes(ax, f"Search convergence — {dataset}", "LLM calls", "Best accuracy so far")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_budget_scatter(df: pd.DataFrame, out_path: Path, dataset: str = "logic") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[df["dataset"] == dataset].copy()
    if sub.empty:
        return
    calls = []
    for _, row in sub.iterrows():
        b = row.get("budget")
        if isinstance(b, dict):
            calls.append(b.get("llm_calls", np.nan))
        else:
            calls.append(np.nan)
    sub["llm_calls"] = calls
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for method, group in sub.groupby("method"):
        ax.scatter(
            group["llm_calls"],
            group["test_acc"],
            label=method,
            color=COLORS.get(method, "#374151"),
            s=42,
            alpha=0.85,
        )
    _style_axes(ax, f"Test accuracy vs evaluation budget — {dataset}", "LLM calls", "Test accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_all(
    results_dir: Path | None = None,
    figures_dir: Path | None = None,
    suffix: str = "",
) -> List[Path]:
    root = _project_root()
    df = load_runs(results_dir)
    scores = load_final_scores(results_dir)
    if df.empty and not scores:
        print("No runs found. Run: python -m src experiment --portfolio")
        return []
    figures_dir = figures_dir or (root / "docs" / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    datasets = sorted(set(df["dataset"].dropna().unique() if not df.empty else []) | set(scores.keys()))
    for ds in datasets:
        bars = figures_dir / f"accuracy_{ds}{suffix}.png"
        conv = figures_dir / f"convergence_{ds}{suffix}.png"
        scat = figures_dir / f"budget_{ds}{suffix}.png"
        plot_accuracy_bars(df, bars, ds, scores=scores)
        max_calls = 100
        try:
            if not df.empty:
                max_calls = int(
                    max(
                        (row.get("budget") or {}).get("llm_calls", 0)
                        for _, row in df[df["dataset"] == ds].iterrows()
                        if isinstance(row.get("budget"), dict)
                    )
                    or 100
                )
        except Exception:
            max_calls = 100
        plot_convergence(df, conv, ds, max_budget=min(max(max_calls, 20), 220))
        plot_budget_scatter(df, scat, ds)
        written.extend([bars, conv, scat])
        print(f"Wrote figures for {ds}{suffix}")
    print(f"Figures saved under {figures_dir}")
    return written


if __name__ == "__main__":
    generate_all()

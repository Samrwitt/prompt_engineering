# src/report.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import json
import matplotlib.pyplot as plt

def save_json(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")

def plot_curve(curve: List[Dict[str, Any]], out_path: str, title: str) -> None:
    if not curve:
        return
    xs = [p["calls"] for p in curve]
    ys = [p["acc"] for p in curve]
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("#LLM calls")
    plt.ylabel("Accuracy")
    plt.title(title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

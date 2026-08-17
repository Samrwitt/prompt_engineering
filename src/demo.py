"""Lightweight demo: one seed, mock LLM, a handful of methods."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from src.config import DATASET_CATALOG, load_run_config
from src.experiment import EvalConfig, run_budgeted_metaheuristic, save_json
from src.methods import build_methods
from src.model import create_llm
from src.prompt import load_blocks, load_jsonl, split_train_test


def main() -> None:
    cfg = load_run_config("fast")
    cfg.backend = "mock"
    cfg.max_data = 12
    cfg.max_llm_calls = 60
    spec = DATASET_CATALOG["logic"]
    blocks = load_blocks(spec["blocks"])
    data = load_jsonl(spec["path"])[: cfg.max_data]
    train, test = split_train_test(data, seed=42)
    oracle = {str(r["q"]): str(r["a"]) for r in train + test}
    llm = create_llm("mock", oracle=oracle)
    eval_cfg = EvalConfig(max_llm_calls=cfg.max_llm_calls, fast_k=5, max_demos=3)
    wanted = {"BASELINE_ALL", "SA+", "GWO", "HYBRID_DE_SA"}
    pack: Dict[str, Any] = {"demo_mode": True, "runs": []}

    print("DEMO (mock LLM) on BBH Boolean Expressions")
    for name, fn, kwargs in build_methods(len(blocks), cfg):
        if name not in wanted:
            continue
        t0 = time.perf_counter()
        report = run_budgeted_metaheuristic(
            method=name,
            algo_fn=fn,
            blocks=blocks,
            demo_candidates=train,
            answer_type=spec["answer_type"],
            llm=llm,
            train_data=train,
            test_data=test,
            seed=0,
            eval_cfg=eval_cfg,
            algo_kwargs=kwargs,
            backend="mock",
        )
        elapsed = time.perf_counter() - t0
        print(f"  {name:14s} test={report.test_acc:.3f}  calls={report.budget.llm_calls}  {elapsed:.1f}s")
        pack["runs"].append({
            "name": name,
            "test_acc": report.test_acc,
            "train_acc": report.train_acc,
            "best_x": report.best_x,
            "instruction": report.best_instruction_text,
        })

    best = max(pack["runs"], key=lambda r: r["test_acc"])
    pack["best"] = best
    Path("results").mkdir(exist_ok=True)
    save_json("results/demo_scores.json", pack)
    print(f"\nBest: {best['name']}  test={best['test_acc']:.3f}")
    print("Saved results/demo_scores.json")


if __name__ == "__main__":
    main()

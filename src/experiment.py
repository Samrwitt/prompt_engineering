"""Budget-equal prompt optimization experiments."""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import RunConfig, load_run_config
from src.methods import build_methods
from src.model import LLMClient, LLMConfig, create_llm
from src.prompt import (
    approx_tokens,
    build_prompt,
    decode_genome,
    extract_answer,
    format_input,
    load_blocks,
    load_jsonl,
    normalize_gt,
    save_json,
    split_train_test,
)


@dataclass
class BudgetStats:
    llm_calls: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    prompt_tokens_est: int = 0
    completion_tokens_est: int = 0

    def add_call(self, prompt: str, completion: str) -> None:
        self.llm_calls += 1
        self.prompt_chars += len(prompt or "")
        self.completion_chars += len(completion or "")
        self.prompt_tokens_est += approx_tokens(prompt or "")
        self.completion_tokens_est += approx_tokens(completion or "")


@dataclass
class EvalConfig:
    max_llm_calls: int
    fast_k: int
    early_stop: bool = True
    max_demos: int = 5


@dataclass
class RunReport:
    dataset: str
    method: str
    seed: int
    train_acc: float
    test_acc: float
    wall_time_sec: float
    budget: BudgetStats
    best_x: List[int]
    best_instruction_text: str = ""
    selected_demo_indices: List[int] = field(default_factory=list)
    curve: List[Tuple[int, float]] = field(default_factory=list)
    backend: str = "mock"


class BudgetedEvaluator:
    """Accuracy under a shared cache, call budget, and early stopping."""

    def __init__(
        self,
        llm: LLMClient,
        blocks: List[str],
        demo_candidates: List[Dict[str, Any]],
        answer_type: str,
        budget: BudgetStats,
        cache: Dict[Tuple[Any, ...], str],
        eval_cfg: EvalConfig,
    ) -> None:
        self.llm = llm
        self.blocks = blocks
        self.demo_candidates = demo_candidates
        self.answer_type = answer_type
        self.budget = budget
        self.cache = cache
        self.eval_cfg = eval_cfg
        self.best_so_far: float = -1.0
        self.curve: List[Tuple[int, float]] = []

    def _call_llm(self, prompt: str, system: str = "") -> str:
        if self.budget.llm_calls >= self.eval_cfg.max_llm_calls:
            return ""
        out = self.llm.generate(prompt, system=system)
        self.budget.add_call(prompt, out)
        return out

    def accuracy(
        self,
        x: List[int],
        dataset: List[Dict[str, Any]],
        current_best: float = -1.0,
        fast: bool = False,
        seed: int = 0,
    ) -> float:
        import random

        rows = dataset
        if fast:
            rng = random.Random(seed)
            idx = list(range(len(dataset)))
            rng.shuffle(idx)
            idx = idx[: max(1, min(self.eval_cfg.fast_k, len(idx)))]
            rows = [dataset[i] for i in idx]

        n_blocks = len(self.blocks)
        n_demos = len(self.demo_candidates)
        x_instr, selected_indices = decode_genome(
            x, n_blocks, n_demos, max_demos=self.eval_cfg.max_demos
        )
        system_prompt = build_prompt(self.blocks, x_instr)
        final_demo_indices = tuple(selected_indices)

        demo_prefix = ""
        for idx in final_demo_indices:
            d = self.demo_candidates[idx]
            demo_prefix += f"Question: {d['q']}\nAnswer: {d['a']}\n\n"

        instr_key = tuple(int(b) for b in x_instr)
        correct = 0
        total = len(rows)

        for i, item in enumerate(rows):
            if self.budget.llm_calls >= self.eval_cfg.max_llm_calls:
                break

            q = str(item["q"])
            gt = normalize_gt(item["a"], self.answer_type)
            key = (instr_key, final_demo_indices, q)
            if key in self.cache:
                out = self.cache[key]
            else:
                inp = demo_prefix + format_input(q, self.answer_type)
                out = self._call_llm(inp, system=system_prompt)
                self.cache[key] = out

            if extract_answer(out, self.answer_type) == gt:
                correct += 1

            if self.eval_cfg.early_stop and current_best >= 0 and total > 0:
                remaining = total - (i + 1)
                if (correct + remaining) / total < current_best:
                    break

        acc = (correct / total) if total else 0.0
        if acc > self.best_so_far:
            self.best_so_far = acc
            self.curve.append((self.budget.llm_calls, self.best_so_far))
        return acc


def run_budgeted_metaheuristic(
    *,
    method: str,
    algo_fn,
    blocks: List[str],
    demo_candidates: List[Dict[str, Any]],
    answer_type: str,
    llm: LLMClient,
    train_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    seed: int,
    eval_cfg: EvalConfig,
    algo_kwargs: Dict[str, Any],
    backend: str = "mock",
) -> RunReport:
    t0 = time.time()
    budget = BudgetStats()
    cache: Dict[Tuple[Any, ...], str] = {}
    search_cfg = EvalConfig(
        max_llm_calls=max(0, eval_cfg.max_llm_calls - len(train_data) - len(test_data)),
        fast_k=eval_cfg.fast_k,
        early_stop=eval_cfg.early_stop,
        max_demos=eval_cfg.max_demos,
    )
    evaluator = BudgetedEvaluator(
        llm, blocks, demo_candidates, answer_type, budget, cache, search_cfg
    )

    best_seen = {"f": -1.0}

    def eval_fn(x: List[int]) -> float:
        f = evaluator.accuracy(x, train_data, current_best=best_seen["f"], fast=True, seed=seed)
        if f > best_seen["f"]:
            best_seen["f"] = f
        return f

    n_dim = len(blocks) + len(demo_candidates)
    best_x, _best_fast, _hist = algo_fn(
        eval_fn=eval_fn,
        n_dim=n_dim,
        seed=seed,
        **algo_kwargs,
    )

    evaluator.eval_cfg = EvalConfig(
        max_llm_calls=eval_cfg.max_llm_calls,
        fast_k=eval_cfg.fast_k,
        early_stop=False,
        max_demos=eval_cfg.max_demos,
    )
    train_acc = evaluator.accuracy(best_x, train_data, current_best=-1.0, fast=False, seed=seed)
    test_acc = evaluator.accuracy(best_x, test_data, current_best=-1.0, fast=False, seed=seed)

    n_blocks = len(blocks)
    _instr, selected_demos = decode_genome(best_x, n_blocks, len(demo_candidates), eval_cfg.max_demos)

    return RunReport(
        dataset="",
        method=method,
        seed=seed,
        train_acc=float(train_acc),
        test_acc=float(test_acc),
        wall_time_sec=float(time.time() - t0),
        budget=budget,
        best_x=[int(b) for b in best_x],
        best_instruction_text=build_prompt(blocks, best_x[:n_blocks]),
        selected_demo_indices=selected_demos,
        curve=evaluator.curve,
        backend=backend,
    )


def summarize_reports(reports: List[RunReport]) -> Dict[str, Any]:
    trains = [r.train_acc for r in reports]
    tests = [r.test_acc for r in reports]
    calls = [r.budget.llm_calls for r in reports]
    tok_in = [r.budget.prompt_tokens_est for r in reports]
    tok_out = [r.budget.completion_tokens_est for r in reports]
    wall = [r.wall_time_sec for r in reports]
    return {
        "mean_train": float(np.mean(trains)),
        "std_train": float(np.std(trains)),
        "mean_test": float(np.mean(tests)),
        "std_test": float(np.std(tests)),
        "mean_llm_calls": float(np.mean(calls)),
        "mean_prompt_tokens_est": float(np.mean(tok_in)),
        "mean_completion_tokens_est": float(np.mean(tok_out)),
        "mean_wall_time_sec": float(np.mean(wall)),
        "runs": [asdict(r) for r in reports],
    }


def save_curves_csv(
    path: str,
    dataset_name: str,
    curves: Dict[str, List[Tuple[int, float]]],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "llm_calls", "best_acc"])
        for method, pts in curves.items():
            for calls, acc in pts:
                w.writerow([dataset_name, method, calls, acc])


def plot_curves_png(
    path: str,
    dataset_name: str,
    curves: Dict[str, List[Tuple[int, float]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for method, pts in curves.items():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, label=method, linewidth=1.8)
    plt.xlabel("# LLM Calls")
    plt.ylabel("Best Accuracy So Far")
    plt.title(f"Accuracy vs Budget ({dataset_name})")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _oracle_from_rows(*row_groups: List[Dict[str, Any]]) -> Dict[str, str]:
    oracle: Dict[str, str] = {}
    for rows in row_groups:
        for item in rows:
            oracle[str(item["q"])] = str(item["a"])
    return oracle


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def run_experiment(cfg: RunConfig) -> Dict[str, Any]:
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_path = results_dir / "runs.jsonl"

    llm_cfg = LLMConfig()
    eval_cfg = EvalConfig(
        max_llm_calls=cfg.max_llm_calls,
        fast_k=cfg.fast_k,
        early_stop=cfg.early_stop,
        max_demos=cfg.max_demos,
    )

    final_scores: Dict[str, Any] = {}
    budget_report: Dict[str, Any] = {}

    print(f">>> {cfg.name.upper()} MODE | backend={cfg.backend} | seeds={cfg.seeds} <<<")

    for ds_name, spec in cfg.dataset_specs():
        print("\n==========================================")
        print(f"Dataset: {ds_name} ({spec['label']})")
        print("==========================================")

        blocks_path, ds_path = spec["blocks"], spec["path"]
        if not Path(blocks_path).exists():
            print(f"Skipping {ds_name}: {blocks_path} not found.")
            continue
        if not Path(ds_path).exists():
            print(f"Skipping {ds_name}: {ds_path} not found.")
            continue

        blocks = load_blocks(blocks_path)
        data = load_jsonl(ds_path)
        if cfg.max_data:
            data = data[: cfg.max_data]
        train_data, test_data = split_train_test(data, train_ratio=0.8, seed=42)
        answer_type = spec["answer_type"]
        llm = create_llm(
            cfg.backend,
            llm_cfg,
            oracle=_oracle_from_rows(train_data, test_data) if cfg.backend == "mock" else None,
        )

        print(f"Blocks: {len(blocks)} | train={len(train_data)} test={len(test_data)}")
        print(f"Budget: max_llm_calls={eval_cfg.max_llm_calls}, fast_k={eval_cfg.fast_k}")

        ds_scores: Dict[str, Any] = {}
        ds_budget: Dict[str, Any] = {}
        curves_for_plot: Dict[str, List[Tuple[int, float]]] = {}

        if not cfg.skip_dspy and cfg.backend != "mock":
            try:
                from src.baseline import run_dspy_miprov2_baseline

                print("Running DSPy MIPROv2 Baseline...")
                t0 = time.time()
                dspy_train, dspy_test = run_dspy_miprov2_baseline(
                    train_rows=train_data,
                    test_rows=test_data,
                    answer_type=answer_type,
                    auto="light",
                    seed=0,
                    base_url=llm_cfg.base_url,
                    max_bootstrapped_demos=1 if cfg.name == "fast" else 3,
                    max_labeled_demos=2 if cfg.name == "fast" else 4,
                )
                wall = time.time() - t0
                ds_scores["dspy_miprov2"] = {"train": float(dspy_train), "test": float(dspy_test)}
                _append_jsonl(runs_path, {
                    "dataset": ds_name,
                    "method": "dspy_miprov2",
                    "seed": 0,
                    "train_acc": float(dspy_train),
                    "test_acc": float(dspy_test),
                    "wall_time_sec": float(wall),
                    "budget": {"llm_calls": 0, "note": "DSPy budget not instrumented"},
                    "best_x": [],
                    "best_instruction_text": "DSPy Optimized Prompt",
                    "selected_demo_indices": [],
                    "backend": cfg.backend,
                })
                ds_budget["dspy_miprov2"] = {
                    "wall_time_sec": float(wall),
                    "llm_calls": None,
                    "note": "DSPy call counting not instrumented in this runner.",
                }
                print(f"  DSPy Train={dspy_train:.3f}, DSPy Test={dspy_test:.3f} (time={wall:.1f}s)")
            except Exception as e:
                print(f"  Skipping DSPy MIPROv2: {e}")
        else:
            print("Skipping DSPy (use --dspy with --backend ollama to enable).")

        methods = build_methods(len(blocks), cfg)
        for method_name, algo_fn, algo_kwargs in methods:
            print(f"Running {method_name} over {len(cfg.seeds)} seed(s)...")
            reports: List[RunReport] = []
            per_seed_curves: List[List[Tuple[int, float]]] = []
            for s in cfg.seeds:
                rep = run_budgeted_metaheuristic(
                    method=method_name,
                    algo_fn=algo_fn,
                    blocks=blocks,
                    demo_candidates=train_data,
                    answer_type=answer_type,
                    llm=llm,
                    train_data=train_data,
                    test_data=test_data,
                    seed=s,
                    eval_cfg=eval_cfg,
                    algo_kwargs=algo_kwargs,
                    backend=cfg.backend,
                )
                rep.dataset = ds_name
                reports.append(rep)
                per_seed_curves.append(rep.curve)
                print(
                    f"  Seed {s}: Train={rep.train_acc:.3f} Test={rep.test_acc:.3f} "
                    f"Calls={rep.budget.llm_calls} Time={rep.wall_time_sec:.1f}s"
                )
                _append_jsonl(runs_path, {**asdict(rep), "dataset": ds_name})

            ds_scores[method_name] = summarize_reports(reports)
            all_pts: List[Tuple[int, float]] = []
            for pts in per_seed_curves:
                all_pts.extend(pts)
            all_pts.sort(key=lambda t: t[0])
            curves_for_plot[method_name] = all_pts

        final_scores[ds_name] = ds_scores
        budget_report[ds_name] = ds_budget
        save_curves_csv(str(results_dir / f"budget_curves_{ds_name}.csv"), ds_name, curves_for_plot)
        plot_curves_png(str(results_dir / f"accuracy_vs_budget_{ds_name}.png"), ds_name, curves_for_plot)

        print(f"\nSummary for {ds_name}:")
        if "dspy_miprov2" in ds_scores:
            print(f"  DSPy MIPROv2 Test: {ds_scores['dspy_miprov2']['test']:.3f}")
        for method_name, *_ in methods:
            m = ds_scores[method_name]
            print(f"  {method_name}: {m['mean_test']:.3f} ± {m['std_test']:.3f} (calls~{m['mean_llm_calls']:.0f})")

    save_json(str(results_dir / "final_scores.json"), final_scores)
    save_json(str(results_dir / "budget_report.json"), budget_report)
    print("\nSaved:")
    print(f"  {results_dir / 'final_scores.json'}")
    print(f"  {results_dir / 'budget_report.json'}")
    print(f"  {results_dir / 'budget_curves_<dataset>.csv'}")
    print(f"  {results_dir / 'accuracy_vs_budget_<dataset>.png'}")
    return final_scores


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint instruction + few-shot prompt optimization with metaheuristics."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="Tiny debug run (default if no mode is set).")
    mode.add_argument("--balanced", action="store_true", help="3 seeds, two datasets.")
    mode.add_argument("--research", action="store_true", help="Full paper-scale run.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path (overrides preset).")
    parser.add_argument("--backend", choices=["mock", "ollama"], default=None)
    parser.add_argument("--max_data", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names.")
    parser.add_argument("--dspy", action="store_true", help="Enable DSPy MIPROv2 (Ollama only).")
    parser.add_argument("--results-dir", type=str, default=None)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    if args.research:
        preset = "research"
    elif args.balanced:
        preset = "balanced"
    else:
        preset = "fast"
    overrides: Dict[str, Any] = {}
    if args.backend:
        overrides["backend"] = args.backend
    if args.max_data is not None:
        overrides["max_data"] = args.max_data
    if args.seeds:
        overrides["seeds"] = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.datasets:
        overrides["datasets"] = [s.strip() for s in args.datasets.split(",") if s.strip()]
    if args.dspy:
        overrides["skip_dspy"] = False
    elif args.backend == "mock" or (args.backend is None and preset == "fast"):
        overrides["skip_dspy"] = True
    if args.results_dir:
        overrides["results_dir"] = args.results_dir
    if preset == "fast" and "backend" not in overrides:
        overrides["backend"] = "mock"
    return load_run_config(preset=preset, config_path=args.config, overrides=overrides)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = config_from_args(args)
    run_experiment(cfg)


if __name__ == "__main__":
    main()

# src/experiment.py
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.model import OllamaLLM, LLMConfig
from src.baseline import run_dspy_miprov2_baseline

# Import your metaheuristics (rename these imports to match your project)
from src.sa import simulated_annealing_sharp
from src.de import differential_evolution_binary_sharp
from src.gwo import grey_wolf_optimizer_binary_sharp
from src.de_sa import hybrid_de_sa


# -----------------------------
# IO helpers
# -----------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_blocks(path: str) -> List[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_train_test(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    n_train = int(len(data) * train_ratio)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    train = [data[i] for i in train_idx]
    test = [data[i] for i in test_idx]
    return train, test


# -----------------------------
# Budget accounting
# -----------------------------

def approx_tokens(text: str) -> int:
    """
    Token estimate that works offline for any model.
    Rule of thumb: ~4 chars/token for English-like text.
    """
    if not text:
        return 0
    return int(math.ceil(len(text) / 4))


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
class RunReport:
    dataset: str
    method: str
    seed: int
    train_acc: float
    test_acc: float
    wall_time_sec: float
    budget: BudgetStats
    best_x: List[int]


# -----------------------------
# Prompt rendering + scoring
# -----------------------------

_NUM_RE = __import__("re").compile(r"-?\d+(?:\.\d+)?")


def build_prompt(blocks: List[str], x: List[int]) -> str:
    chosen = [b for b, bit in zip(blocks, x) if int(bit) == 1]
    return "\n".join(chosen).strip()


def extract_answer(text: str, answer_type: str) -> str:
    import re
    t = (text or "").strip().lower()

    if answer_type == "yesno":
        if re.search(r"\byes\b", t):
            return "yes"
        if re.search(r"\bno\b", t):
            return "no"
        if re.search(r"\btrue\b", t):
            return "yes"
        if re.search(r"\bfalse\b", t):
            return "no"
        if re.search(r"\b1\b", t):
            return "yes"
        if re.search(r"\b0\b", t):
            return "no"
        return ""

    if answer_type == "abcd":
        m = re.search(r"\b([abcd])\b", t)
        return m.group(1) if m else ""

    if answer_type == "number":
        m = re.search(r"answer[^0-9-]*(-?\d+(?:\.\d+)?)", t)
        if m:
            return m.group(1)
        nums = _NUM_RE.findall(t.replace(",", ""))
        return nums[-1] if nums else ""

    return t


def normalize_gt(a_raw: Any, answer_type: str) -> str:
    a = str(a_raw).strip().lower()
    if answer_type == "yesno":
        if a in ("1", "yes", "true"):
            return "yes"
        if a in ("0", "no", "false"):
            return "no"
        return a
    if answer_type == "abcd":
        return a[:1]
    return a


def format_input(prompt_prefix: str, q: str, answer_type: str) -> str:
    if answer_type == "yesno":
        return f"{prompt_prefix}\n\nQuestion: {q}\nAnswer (yes or no): "
    if answer_type == "abcd":
        return f"{prompt_prefix}\n\nQuestion: {q}\nAnswer (A/B/C/D): "
    return f"{prompt_prefix}\n\nQuestion: {q}\nAnswer (integer only): "


# -----------------------------
# Budgeted evaluator with early stop + caching
# -----------------------------

@dataclass
class EvalConfig:
    max_llm_calls: int
    fast_k: int  # number of train items used for "fast fitness"
    early_stop: bool = True


class BudgetedEvaluator:
    """
    Evaluates accuracy under:
      - shared cache: (x_tuple, question) -> model_output
      - budget tracking (calls/tokens)
      - early stopping vs current_best
      - curve logging: best_acc vs llm_calls
    """

    def __init__(
        self,
        llm: OllamaLLM,
        blocks: List[str],
        answer_type: str,
        budget: BudgetStats,
        cache: Dict[Tuple[Tuple[int, ...], str], str],
        eval_cfg: EvalConfig,
    ) -> None:
        self.llm = llm
        self.blocks = blocks
        self.answer_type = answer_type
        self.budget = budget
        self.cache = cache
        self.eval_cfg = eval_cfg

        self.best_so_far: float = -1.0
        self.curve: List[Tuple[int, float]] = []  # (llm_calls, best_acc)

    def _call_llm(self, prompt: str) -> str:
        if self.budget.llm_calls >= self.eval_cfg.max_llm_calls:
            # Hard budget stop: no more calls allowed
            return ""
        out = self.llm.generate(prompt)
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
        """
        If fast=True, evaluates only first K items (after a deterministic shuffle).
        Early stop: if even perfect remaining can't beat current_best.
        """
        import random

        rows = dataset
        if fast:
            rng = random.Random(seed)
            idx = list(range(len(dataset)))
            rng.shuffle(idx)
            idx = idx[: max(1, min(self.eval_cfg.fast_k, len(idx)))]
            rows = [dataset[i] for i in idx]

        prompt_prefix = build_prompt(self.blocks, x)
        x_key = tuple(int(b) for b in x)

        correct = 0
        total = len(rows)

        for i, item in enumerate(rows):
            # If budget exceeded, stop (returns partial accuracy so optimizers still progress)
            if self.budget.llm_calls >= self.eval_cfg.max_llm_calls:
                break

            q = str(item["q"])
            gt = normalize_gt(item["a"], self.answer_type)

            key = (x_key, q)
            if key in self.cache:
                out = self.cache[key]
            else:
                inp = format_input(prompt_prefix, q, self.answer_type)
                out = self._call_llm(inp)
                self.cache[key] = out

            pred = extract_answer(out, self.answer_type)
            if pred == gt:
                correct += 1

            # Early stop if cannot beat current_best anymore
            if self.eval_cfg.early_stop and current_best >= 0 and total > 0:
                remaining = total - (i + 1)
                best_possible = (correct + remaining) / total
                if best_possible < current_best:
                    break

        acc = (correct / total) if total else 0.0

        if acc > self.best_so_far:
            self.best_so_far = acc
            self.curve.append((self.budget.llm_calls, self.best_so_far))

        return acc


# -----------------------------
# Method runners (budget-equal)
# -----------------------------

def run_budgeted_metaheuristic(
    *,
    method: str,
    algo_fn,
    blocks: List[str],
    answer_type: str,
    llm: OllamaLLM,
    train_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    seed: int,
    eval_cfg: EvalConfig,
    algo_kwargs: Dict[str, Any],
) -> RunReport:
    t0 = time.time()

    budget = BudgetStats()
    cache: Dict[Tuple[Tuple[int, ...], str], str] = {}
    evaluator = BudgetedEvaluator(llm, blocks, answer_type, budget, cache, eval_cfg)

    # Fitness function used by optimizer (FAST)
    def fast_eval(x: List[int], current_best: float = -1.0) -> float:
        return evaluator.accuracy(x, train_data, current_best=current_best, fast=True, seed=seed)

    # Wrap to give optimizers a plain eval_fn signature
    # but we still want early-stop vs "best seen by optimizer".
    best_seen = {"f": -1.0}

    def eval_fn(x: List[int]) -> float:
        f = fast_eval(x, current_best=best_seen["f"])
        if f > best_seen["f"]:
            best_seen["f"] = f
        return f

    # Run optimization (under same eval function + same budget cap)
    best_x, best_fast, _curve = algo_fn(
        eval_fn=eval_fn,
        n_dim=len(blocks),
        seed=seed,
        **algo_kwargs,
    )

    # Full train + full test (still under remaining budget, but uses cache heavily)
    train_acc = evaluator.accuracy(best_x, train_data, current_best=-1.0, fast=False, seed=seed)
    test_acc = evaluator.accuracy(best_x, test_data, current_best=-1.0, fast=False, seed=seed)

    wall = time.time() - t0

    return RunReport(
        dataset="",
        method=method,
        seed=seed,
        train_acc=float(train_acc),
        test_acc=float(test_acc),
        wall_time_sec=float(wall),
        budget=budget,
        best_x=[int(b) for b in best_x],
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


# -----------------------------
# Plotting curves
# -----------------------------

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
            for (calls, acc) in pts:
                w.writerow([dataset_name, method, calls, acc])


def plot_curves_png(
    path: str,
    dataset_name: str,
    curves: Dict[str, List[Tuple[int, float]]],
) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    for method, pts in curves.items():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, label=method)

    plt.xlabel("# LLM Calls")
    plt.ylabel("Best Accuracy So Far")
    plt.title(f"Accuracy vs Budget ({dataset_name})")
    plt.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main experiment
# -----------------------------

def main() -> None:
    datasets_to_run = [
        ("arithmetic", "data/arithmetic.jsonl"),
        ("logic", "data/logic.jsonl"),
    ]

    cfg = LLMConfig()
    llm = OllamaLLM(cfg)

    SEEDS = [0, 1, 2, 3, 4]

    # FAIR budget per method per seed (tune this!)
    # This is the main “research knob”.
    EVAL_CFG = EvalConfig(
        max_llm_calls=600,   # total calls allowed per (method, seed, dataset)
        fast_k=16,           # "fast fitness" subset size
        early_stop=True,
    )

    final_scores: Dict[str, Any] = {}
    budget_report: Dict[str, Any] = {}

    for ds_name, ds_path in datasets_to_run:
        print("\n==========================================")
        print(f"Running on Dataset: {ds_name}")
        print("==========================================")

        if ds_name in ("toy_math", "arithmetic"):
            answer_type = "number"
            blocks_path = "prompts/blocks_number.json"
        else:
            answer_type = "yesno"
            blocks_path = "prompts/blocks_yesno.json"

        if not Path(blocks_path).exists():
            print(f"Skipping {ds_name}: {blocks_path} not found.")
            continue
        if not Path(ds_path).exists():
            print(f"Skipping {ds_name}: {ds_path} not found.")
            continue

        blocks = load_blocks(blocks_path)
        data = load_jsonl(ds_path)
        train_data, test_data = split_train_test(data, train_ratio=0.8, seed=42)

        print(f"Blocks: {blocks_path} (n={len(blocks)}) | answer_type={answer_type}")
        print(f"Data split: {len(train_data)} train, {len(test_data)} test")
        print(f"Fair budget per run: max_llm_calls={EVAL_CFG.max_llm_calls}, fast_k={EVAL_CFG.fast_k}")

        ds_scores: Dict[str, Any] = {}
        ds_budget: Dict[str, Any] = {}
        curves_for_plot: Dict[str, List[Tuple[int, float]]] = {}

        # ---------- DSPy baseline ----------
        # NOTE: DSPy uses litellm under the hood; counting exact LLM calls requires hooking litellm.
        # We still record train/test and wall-clock. Budget fields are set to None.
        try:
            print("Running DSPy MIPROv2 Baseline...")
            t0 = time.time()
            dspy_train, dspy_test = run_dspy_miprov2_baseline(
                train_rows=train_data,
                test_rows=test_data,
                answer_type=answer_type,
                auto="light",
                seed=0,
                base_url=cfg.base_url,
            )
            wall = time.time() - t0
            ds_scores["dspy_miprov2"] = {"train": float(dspy_train), "test": float(dspy_test)}
            ds_budget["dspy_miprov2"] = {
                "wall_time_sec": float(wall),
                "llm_calls": None,
                "tokens_in_est": None,
                "tokens_out_est": None,
                "note": "DSPy call counting not instrumented in this runner.",
            }
            print(f"  DSPy Train={dspy_train:.3f}, DSPy Test={dspy_test:.3f} (time={wall:.1f}s)")
        except Exception as e:
            print(f"  Skipping DSPy MIPROv2: {e}")

        # ---------- Metaheuristics under fair budget ----------
        methods = [
            ("SA+", simulated_annealing_sharp, dict(iters=60, t0=1.0, cooling=0.97, stagnation_reheat=12)),
            ("DE", differential_evolution_binary_sharp, dict(pop_size=18, iters=25, F=0.7, CR=0.8)),
            ("GWO", grey_wolf_optimizer_binary_sharp, dict(pack_size=18, iters=25)),
            ("HYBRID_DE_SA", hybrid_de_sa, dict(
                pop_size=18, de_iters=20,
                sa_iters=40,
            )),
        ]

        for method_name, algo_fn, algo_kwargs in methods:
            print(f"Running {method_name} over {len(SEEDS)} seeds...")
            reports: List[RunReport] = []
            # For plotting: aggregate curves across seeds by taking envelope later
            per_seed_curves: List[List[Tuple[int, float]]] = []

            for s in SEEDS:
                rep = run_budgeted_metaheuristic(
                    method=method_name,
                    algo_fn=algo_fn,
                    blocks=blocks,
                    answer_type=answer_type,
                    llm=llm,
                    train_data=train_data,
                    test_data=test_data,
                    seed=s,
                    eval_cfg=EVAL_CFG,
                    algo_kwargs=algo_kwargs,
                )
                rep.dataset = ds_name
                reports.append(rep)

                # Re-run curve extraction by rebuilding evaluator? We stored curve internally only.
                # So: we reconstruct a minimal curve from budget usage:
                # -> we store at least one point: (calls, best_test) as a fallback.
                # If you want full curves per run, add curve export in run_budgeted_metaheuristic.
                per_seed_curves.append([(rep.budget.llm_calls, rep.test_acc)])

                print(
                    f"  Seed {s}: Train={rep.train_acc:.3f} Test={rep.test_acc:.3f} "
                    f"Calls={rep.budget.llm_calls} Tok~in={rep.budget.prompt_tokens_est} Tok~out={rep.budget.completion_tokens_est} "
                    f"Time={rep.wall_time_sec:.1f}s"
                )

            ds_scores[method_name] = summarize_reports(reports)

            # Build a simple “best-vs-budget” curve across seeds:
            # Here we only have one point per seed (end-of-run). If you want detailed curves,
            # add `curve` to RunReport and append evaluator.curve inside runner.
            # Still publishable if you present “final accuracy at fixed budget”.
            # But: we also write CSV and PNG; they’ll be flat with current settings.
            # Upgrade path: export evaluator.curve per run.
            all_pts: List[Tuple[int, float]] = []
            for pts in per_seed_curves:
                all_pts.extend(pts)
            all_pts.sort(key=lambda t: t[0])
            curves_for_plot[method_name] = all_pts

        final_scores[ds_name] = ds_scores
        budget_report[ds_name] = ds_budget

        # Save per-dataset curves
        save_curves_csv(
            f"results/budget_curves_{ds_name}.csv",
            ds_name,
            curves_for_plot,
        )
        plot_curves_png(
            f"results/accuracy_vs_budget_{ds_name}.png",
            ds_name,
            curves_for_plot,
        )

        print(f"\nSummary for {ds_name}:")
        if "dspy_miprov2" in ds_scores:
            print(f"  DSPy MIPROv2 Test: {ds_scores['dspy_miprov2']['test']:.3f}")
        for method_name, *_ in methods:
            m = ds_scores[method_name]
            print(f"  {method_name}: {m['mean_test']:.3f} ± {m['std_test']:.3f} (calls~{m['mean_llm_calls']:.0f})")

    save_json("results/final_scores.json", final_scores)
    save_json("results/budget_report.json", budget_report)
    print("\nSaved:")
    print("  results/final_scores.json")
    print("  results/budget_report.json")
    print("  results/budget_curves_<dataset>.csv")
    print("  results/accuracy_vs_budget_<dataset>.png")


if __name__ == "__main__":
    main()

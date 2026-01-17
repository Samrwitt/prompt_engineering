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

from src.simple_opt import random_search, greedy_add_one, hill_climb_bit_flip


# -----------------------------
# IO helpers
# -----------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//") and not line.startswith("#"):
                try:
                    obj = json.loads(line)
                    # Map GSM8K or other formats to q/a
                    if "question" in obj and "q" not in obj:
                        obj["q"] = obj["question"]
                    if "answer" in obj and "a" not in obj:
                        obj["a"] = obj["answer"]
                    rows.append(obj)
                except json.JSONDecodeError:
                    continue
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
    best_instruction_text: str = ""
    selected_demo_indices: List[int] = None


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


def format_input(q: str, answer_type: str) -> str:
    if answer_type == "yesno":
        return f"Question: {q}\nAnswer (yes or no): "
    if answer_type == "abcd":
        return f"Question: {q}\nAnswer (A/B/C/D): "
    return f"Question: {q}\nAnswer (integer only): "


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
        demo_candidates: List[Dict[str, Any]],
        answer_type: str,
        budget: BudgetStats,
        cache: Dict[Tuple[Tuple[int, ...], str], str],
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
        self.curve: List[Tuple[int, float]] = []  # (llm_calls, best_acc)

    def _call_llm(self, prompt: str, system: str = "") -> str:
        if self.budget.llm_calls >= self.eval_cfg.max_llm_calls:
            # Hard budget stop: no more calls allowed
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

        # Parse x: [x_instr (len=n_blocks), x_demo (len=n_demos)]
        n_blocks = len(self.blocks)
        n_demos = len(self.demo_candidates)
        
        if len(x) < n_blocks:
            x_instr = x
            x_demo = []
        else:
            x_instr = x[:n_blocks]
            x_demo = x[n_blocks : n_blocks + n_demos]

        system_prompt = build_prompt(self.blocks, x_instr)
        
        # Build few-shot prefix + stable cache key components
        MAX_DEMOS = 5
        selected_indices = [i for i, bit in enumerate(x_demo) if int(bit) == 1]
        final_demo_indices = tuple(selected_indices[:MAX_DEMOS])
        
        demo_prefix = ""
        for idx in final_demo_indices:
            d = self.demo_candidates[idx]
            demo_prefix += f"Question: {d['q']}\nAnswer: {d['a']}\n\n"

        # Stable cache key: (instr_bits, demo_indices, dataset_q)
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
                final_q_block = format_input(q, self.answer_type)
                inp = demo_prefix + final_q_block
                out = self._call_llm(inp, system=system_prompt)
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
    demo_candidates: List[Dict[str, Any]],
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
    evaluator = BudgetedEvaluator(llm, blocks, demo_candidates, answer_type, budget, cache, eval_cfg)

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

    # Decision dimension = Instructions + Candidates
    n_dim = len(blocks) + len(demo_candidates)

    # Reserve budget for final evaluations: len(train) + len(test)
    # Note: Evaluator stop will kick in. 
    # But we want the algorithm to know the search_budget
    total_eval_budget = len(train_data) + len(test_data)
    search_limit = max(0, eval_cfg.max_llm_calls - total_eval_budget)
    
    # Temporarily restrict evaluator for search
    orig_max = eval_cfg.max_llm_calls
    eval_cfg.max_llm_calls = search_limit

    # Run optimization
    best_x, best_fast, _hist = algo_fn(
        eval_fn=eval_fn,
        n_dim=n_dim,
        seed=seed,
        **algo_kwargs,
    )
    
    # Restore budget for final evaluation
    eval_cfg.max_llm_calls = orig_max

    # Full train + full test (still under remaining budget, uses cache heavily)
    train_acc = evaluator.accuracy(best_x, train_data, current_best=-1.0, fast=False, seed=seed)
    test_acc = evaluator.accuracy(best_x, test_data, current_best=-1.0, fast=False, seed=seed)

    wall = time.time() - t0

    # Extract demo indices for reporting
    n_blocks = len(blocks)
    x_demo = best_x[n_blocks:] if len(best_x) > n_blocks else []
    selected_demos = [i for i, b in enumerate(x_demo) if int(b) == 1][:5]

    return RunReport(
        dataset="",
        method=method,
        seed=seed,
        train_acc=float(train_acc),
        test_acc=float(test_acc),
        wall_time_sec=float(wall),
        budget=budget,
        best_x=[int(b) for b in best_x],
        best_instruction_text=build_prompt(blocks, best_x[:n_blocks]),
        selected_demo_indices=selected_demos,
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
    import matplotlib
    matplotlib.use("Agg")  # Force non-interactive backend
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
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run in fast debug mode (cleaner logic)")
    parser.add_argument("--research", action="store_true", help="Run in full research mode")
    parser.add_argument("--balanced", action="store_true", help="Run in balanced mode (middle ground)")
    parser.add_argument("--max_data", type=int, default=None, help="Limit number of items per dataset")
    args = parser.parse_args()

    use_fast = args.fast
    use_balanced = args.balanced
    max_data = args.max_data
    
    if use_fast:
        print(">>> FAST MODE: 1 seed, max_data=10, budget=100. <<<")
        run_max_data = 10
        RUN_CFG = {
            "max_llm_calls": 100,
            "fast_k": 5,
            "seeds": [0],
            "sa_iters": 20,
            "pop_size": 8,
            "de_iters": 10,
            "gwo_iters": 10,
            "hybrid_de": 10,
            "hybrid_sa": 10,
        }
    elif use_balanced:
        print(">>> BALANCED MODE: 3 seeds, max_data=20, budget=300. <<<")
        run_max_data = 20
        RUN_CFG = {
            "max_llm_calls": 300,
            "fast_k": 10,
            "seeds": [0, 1, 2],
            "sa_iters": 30,
            "pop_size": 12,
            "de_iters": 15,
            "gwo_iters": 15,
            "hybrid_de": 10,
            "hybrid_sa": 20,
        }
    else:
        print(">>> RESEARCH MODE: 10 seeds, max_data=100, budget=1000. <<<")
        run_max_data = 100
        RUN_CFG = {
            "max_llm_calls": 1000,
            "fast_k": 20,
            "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "sa_iters": 40,
            "pop_size": 20,
            "de_iters": 30,
            "gwo_iters": 30,
            "hybrid_de": 20,
            "hybrid_sa": 40,
        }

    # Override max_data if manually provided
    if max_data is not None:
        run_max_data = max_data

    dataset_list = [
        #("arithmetic", "data/arithmetic.jsonl"),
        ("logic", "data/bbh_boolean_expressions.jsonl"),
        ("gsm8k", "data/gsm8k_sample.jsonl"),
    ]
    
    # In fast mode, maybe only run one dataset? Let's run both but quickly.
    
    cfg = LLMConfig()
    llm = OllamaLLM(cfg)

    SEEDS = RUN_CFG["seeds"]

    EVAL_CFG = EvalConfig(
        max_llm_calls=RUN_CFG["max_llm_calls"],
        fast_k=RUN_CFG["fast_k"],
        early_stop=True,
    )

    final_scores: Dict[str, Any] = {}
    budget_report: Dict[str, Any] = {}

    for ds_name, ds_path in dataset_list:
        print("\n==========================================")
        print(f"Running on Dataset: {ds_name} | Mode: {'FAST' if use_fast else 'RESEARCH'}")
        print("==========================================")

        if ds_name in ("toy_math", "arithmetic"):
            answer_type = "number"
            blocks_path = "prompts/instruction_blocks_number.json"
        else:
            answer_type = "yesno"
            blocks_path = "prompts/instruction_blocks_yesno.json"

        if not Path(blocks_path).exists():
            print(f"Skipping {ds_name}: {blocks_path} not found.")
            continue
        if not Path(ds_path).exists():
            print(f"Skipping {ds_name}: {ds_path} not found.")
            continue

        blocks = load_blocks(blocks_path)
        data = load_jsonl(ds_path)
        
        if run_max_data:
            print(f"Limiting dataset to first {run_max_data} items.")
            data = data[:run_max_data]
            
        train_data, test_data = split_train_test(data, train_ratio=0.8, seed=42)

        print(f"Blocks: {blocks_path} (n={len(blocks)}) | answer_type={answer_type}")
        print(f"Data split: {len(train_data)} train, {len(test_data)} test")
        print(f"Fair budget per run: max_llm_calls={EVAL_CFG.max_llm_calls}, fast_k={EVAL_CFG.fast_k}")

        ds_scores: Dict[str, Any] = {}
        ds_budget: Dict[str, Any] = {}
        curves_for_plot: Dict[str, List[Tuple[int, float]]] = {}

        # ---------- DSPy baseline ----------
        try:
            print("Running DSPy MIPROv2 Baseline...")
            if use_fast:
                print("  (Fast mode: DSPy might still be slow, effectively skipping heavy opt or relying on baseline defaults)")
                # We could potentially pass a 'fast' flag to baseline if it supported it, 
                # but for now we'll just run it. It usually respects some internal defaults.
                # Or we can skip it in fast mode? Let's run it but warn.
            
            t0 = time.time()
            dspy_train, dspy_test = run_dspy_miprov2_baseline(
                train_rows=train_data,
                test_rows=test_data,
                answer_type=answer_type,
                auto="light",
                seed=0,
                base_url=cfg.base_url,
                max_bootstrapped_demos=1 if use_fast else 3,
                max_labeled_demos=2 if use_fast else 4,
            )
            wall = time.time() - t0
            ds_scores["dspy_miprov2"] = {"train": float(dspy_train), "test": float(dspy_test)}
            
            # Log to runs.jsonl for unified analysis
            _dspy_log = {
                "dataset": ds_name,
                "method": "dspy_miprov2",
                "seed": 0,
                "train_acc": float(dspy_train),
                "test_acc": float(dspy_test),
                "wall_time_sec": float(wall),
                "budget": {"llm_calls": 0, "note": "DSPy budget not instrumented"},
                "best_x": [],
                "best_instruction_text": "DSPy Optimized Prompt",
                "selected_demo_indices": []
            }
            with open("results/runs.jsonl", "a", encoding="utf-8") as _f:
                _f.write(json.dumps(_dspy_log) + "\n")

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

        # Use training set as demo candidates.
        # Warning: if training set is large, n_dim grows. 
        # But we want to select *from* the train set.
        demo_candidates = train_data 
        
        # ---------- Baselines (Fixed) ----------
        def fixed_baseline(eval_fn, n_dim, seed, x_fixed, **kwargs):
            # If x_fixed provided is short, pad with 0s? 
            # Or just assume caller provides correct length? 
            # Let's pad it to n_dim to be safe.
            full_x = x_fixed[:]
            if len(full_x) < n_dim:
                full_x += [0] * (n_dim - len(full_x))
            val = eval_fn(full_x)
            return full_x, val, [val]

        # ---------- Metaheuristics matches ----------
        # Note: Fixed baselines need to pad their vectors for the new demo space (0s = no demos)
        ones_blocks = [1] * len(blocks)
        zeros_blocks = [0] * len(blocks)
        
        methods = [
            ("BASELINE_ALL", fixed_baseline, dict(x_fixed=ones_blocks)),
            ("BASELINE_NONE", fixed_baseline, dict(x_fixed=zeros_blocks)),
            # New Simple Baselines
            ("RANDOM", random_search, dict(iters=min(50, RUN_CFG["max_llm_calls"]))), 
            ("GREEDY", greedy_add_one, dict(restarts=1)),
            
            # Metaheuristics
            ("SA+", simulated_annealing_sharp, dict(
                iters=RUN_CFG["sa_iters"], t0=1.0, cooling=0.97, stagnation_reheat=12
            )),
            ("DE", differential_evolution_binary_sharp, dict(
                pop_size=RUN_CFG["pop_size"], iters=RUN_CFG["de_iters"], F=0.7, CR=0.8
            )),
            ("GWO", grey_wolf_optimizer_binary_sharp, dict(
                pack_size=RUN_CFG["pop_size"], iters=RUN_CFG["gwo_iters"]
            )),
            ("HYBRID_DE_SA", hybrid_de_sa, dict(
                pop_size=RUN_CFG["pop_size"], 
                de_iters=RUN_CFG["hybrid_de"], 
                sa_iters=RUN_CFG["hybrid_sa"]
            )),
        ]

        for method_name, algo_fn, algo_kwargs in methods:
            print(f"Running {method_name} over {len(SEEDS)} seeds...")
            reports: List[RunReport] = []
            per_seed_curves: List[List[Tuple[int, float]]] = []

            for s in SEEDS:
                rep = run_budgeted_metaheuristic(
                    method=method_name,
                    algo_fn=algo_fn,
                    blocks=blocks,
                    demo_candidates=demo_candidates,
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
                per_seed_curves.append([(rep.budget.llm_calls, rep.test_acc)])

                print(
                    f"  Seed {s}: Train={rep.train_acc:.3f} Test={rep.test_acc:.3f} "
                    f"Calls={rep.budget.llm_calls} "
                    f"Time={rep.wall_time_sec:.1f}s"
                )
                
                # Log to runs.jsonl
                _log_entry = asdict(rep)
                _log_entry["dataset"] = ds_name # explicit set
                with open("results/runs.jsonl", "a", encoding="utf-8") as _f:
                    _f.write(json.dumps(_log_entry) + "\n")

            ds_scores[method_name] = summarize_reports(reports)

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

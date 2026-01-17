from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.model import OllamaLLM, LLMConfig
from src.fitness import load_dataset_jsonl, load_blocks, PromptEvaluator, split_train_test

# import your "best" methods
from src.baseline import run_dspy_miprov2_baseline
from src.sa import simulated_annealing_sharp
from src.de import differential_evolution_binary_sharp
from src.gwo import grey_wolf_optimizer_binary_sharp
from src.de_sa import hybrid_de_sa


def save_json(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _now() -> float:
    return time.perf_counter()


def _demo_budget_kwargs(n_blocks: int) -> Dict[str, Dict[str, Any]]:
    """
    Tiny budgets that still show algorithm behavior.
    Tune if your laptop is slower/faster.
    """
    return {
        "sa_plus": dict(iters=18, t0=1.0, cooling=0.97, stagnation_reheat=8),
        "de": dict(pop_size=min(10, max(6, n_blocks * 2)), iters=12, F=0.7, CR=0.85),
        "gwo": dict(pack_size=min(10, max(6, n_blocks * 2)), iters=12),
        "hybrid": dict(
            pop_size=min(10, max(6, n_blocks * 2)),
            de_iters=10,
            F=0.7,
            CR=0.85,
            sa_iters=14,
            t0=1.0,
            cooling=0.96,
            flips_per_move=1,
        ),
    }


def _run_one(
    name: str,
    algo_fn,
    evaluator_train: PromptEvaluator,
    evaluator_test: PromptEvaluator,
    seed: int,
    **kwargs
) -> Dict[str, Any]:
    t0 = _now()
    best_x, best_fast, curve = algo_fn(
        eval_fn=evaluator_train.eval_accuracy,
        n_dim=len(evaluator_train.blocks),
        seed=seed,
        **kwargs
    )
    t1 = _now()

    # full train + test on the chosen x (still small in demo mode)
    train_acc = evaluator_train.eval_accuracy(best_x)
    test_acc = evaluator_test.eval_accuracy(best_x)
    t2 = _now()

    return {
        "name": name,
        "seed": seed,
        "best_x": best_x,
        "fast_acc": float(best_fast),
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "curve": [float(c) for c in curve],
        "time_search_s": float(t1 - t0),
        "time_eval_s": float(t2 - t1),
        "time_total_s": float(t2 - t0),
    }


def main() -> None:
    # -----------------------
    # DEMO SETTINGS (FAST!)
    # -----------------------
    DEMO_SEED = 0

    # small slices so it finishes quickly
    TRAIN_SLICE = 16   # fast eval set
    TEST_SLICE  = 10

    # (optional) even faster: only run one dataset during demo
    datasets_to_run = [
        ("logic", "data/logic.jsonl"),
        # ("arithmetic", "data/arithmetic.jsonl"),
    ]

    # Make decoding faster for demo
    cfg = LLMConfig()
    cfg.temperature = 0.0
    cfg.num_predict = 32   # smaller => faster (demo)
    cfg.num_ctx = 256

    llm = OllamaLLM(cfg)

    # Warm-up call so first real call doesn’t “stall”
    _ = llm.generate("Warmup: reply with OK only.")

    final: Dict[str, Any] = {"demo_mode": True, "runs": {}}

    for ds_name, ds_path in datasets_to_run:
        print("\n==========================================")
        print(f"DEMO RUN on Dataset: {ds_name}")
        print("==========================================")

        if ds_name in ("logic", "comparative"):
            answer_type = "yesno"
            blocks_path = "prompts/blocks_yesno.json"
        else:
            answer_type = "number"
            blocks_path = "prompts/blocks_number.json"

        if not Path(blocks_path).exists():
            print(f"Missing {blocks_path}. Skipping.")
            continue
        if not Path(ds_path).exists():
            print(f"Missing {ds_path}. Skipping.")
            continue

        blocks = load_blocks(blocks_path)
        data = load_dataset_jsonl(ds_path)
        train_data, test_data = split_train_test(data, train_ratio=0.8, seed=42)

        # demo slices
        train_demo = train_data[:TRAIN_SLICE]
        test_demo = test_data[:TEST_SLICE]

        train_eval = PromptEvaluator(llm, train_demo, blocks, answer_type=answer_type)
        test_eval  = PromptEvaluator(llm, test_demo, blocks, answer_type=answer_type)

        budgets = _demo_budget_kwargs(len(blocks))

        # 1) DSPy baseline (optional; can be slow on some machines)
        # For live demo: it’s okay to SKIP and just show metaheuristics.
        dspy_result = None
        try:
            print("Running DSPy MIPROv2 (DEMO)...")
            t0 = _now()
            dspy_train, dspy_test = run_dspy_miprov2_baseline(
                train_rows=train_demo,
                test_rows=test_demo,
                answer_type=answer_type,
                auto="light",
                seed=DEMO_SEED,
                base_url=cfg.base_url,
                max_bootstrapped_demos=2,
                max_labeled_demos=2,
            )
            t1 = _now()
            dspy_result = {
                "train_acc": float(dspy_train),
                "test_acc": float(dspy_test),
                "time_total_s": float(t1 - t0),
            }
            print(f"  DSPy Train={dspy_train:.3f} Test={dspy_test:.3f} ({t1-t0:.1f}s)")
        except Exception as e:
            print(f"  DSPy skipped (demo-safe): {e}")

        # 2) Metaheuristics (fast)
        print("Running SA+ (DEMO)...")
        sa_out = _run_one("SA+", simulated_annealing_sharp, train_eval, test_eval, DEMO_SEED, **budgets["sa_plus"])
        print(f"  SA+ Test={sa_out['test_acc']:.3f}  Total={sa_out['time_total_s']:.1f}s")

        print("Running DE (DEMO)...")
        de_out = _run_one("DE", differential_evolution_binary_sharp, train_eval, test_eval, DEMO_SEED, **budgets["de"])
        print(f"  DE  Test={de_out['test_acc']:.3f}  Total={de_out['time_total_s']:.1f}s")

        print("Running GWO (DEMO)...")
        gwo_out = _run_one("GWO", grey_wolf_optimizer_binary_sharp, train_eval, test_eval, DEMO_SEED, **budgets["gwo"])
        print(f"  GWO Test={gwo_out['test_acc']:.3f}  Total={gwo_out['time_total_s']:.1f}s")

        print("Running HYBRID DE->SA (DEMO)...")
        hy_out = _run_one("DE->SA", hybrid_de_sa, train_eval, test_eval, DEMO_SEED, **budgets["hybrid"])
        print(f"  HYB Test={hy_out['test_acc']:.3f}  Total={hy_out['time_total_s']:.1f}s")

        # summarize
        runs = [sa_out, de_out, gwo_out, hy_out]
        best = max(runs, key=lambda r: r["test_acc"])

        ds_pack = {
            "dataset": ds_name,
            "answer_type": answer_type,
            "n_blocks": len(blocks),
            "train_size_demo": len(train_demo),
            "test_size_demo": len(test_demo),
            "dspy": dspy_result,
            "runs": runs,
            "best_demo": {"name": best["name"], "test_acc": best["test_acc"], "x": best["best_x"]},
        }

        final["runs"][ds_name] = ds_pack

        print("\nDEMO SUMMARY:")
        if dspy_result:
            print(f"  DSPy Test: {dspy_result['test_acc']:.3f}")
        print(f"  Best Metaheuristic: {best['name']}  Test={best['test_acc']:.3f}")
        print(f"  Best x: {best['best_x']}")

    save_json("results/demo_scores.json", final)
    print("\nSaved demo results to results/demo_scores.json")


if __name__ == "__main__":
    main()

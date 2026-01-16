# experiment.py (or your main runner file)
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import sys

# Ensure project root is in path so we can import 'src'
sys.path.append(str(Path(__file__).parent.parent))

from src.model import OllamaLLM, LLMConfig
from src.fitness import load_dataset_jsonl, load_blocks, PromptEvaluator, split_train_test
from src.baseline import (
    run_baseline,
    run_random_baseline,
    run_greedy_baseline,
    run_dspy_miprov2_baseline,
)
from src.sa import simulated_annealing
from src.ga import genetic_algorithm
from src.pso import binary_pso


def save_json(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def run_multi_seed_algorithm(
    algo_name: str,
    algo_fn,
    evaluator_train: PromptEvaluator,
    evaluator_test: PromptEvaluator,
    seeds: List[int],
    **algo_kwargs
) -> Dict[str, Any]:
    test_scores = []
    train_scores = []
    best_configs = []

    print(f"Running {algo_name} over {len(seeds)} seeds...")

    for s in seeds:
        print(f"  > Seed {s}...", end="", flush=True)

        # Heuristics return: (best_x, best_f, curve)
        best_x, trains_acc, _ = algo_fn(
            eval_fn=evaluator_train.eval_accuracy,
            n_dim=len(evaluator_train.blocks),
            seed=s,
            **algo_kwargs
        )

        test_acc = evaluator_test.eval_accuracy(best_x)

        train_scores.append(trains_acc)
        test_scores.append(test_acc)
        best_configs.append(best_x)
        print(f" Train={trains_acc:.3f}, Test={test_acc:.3f}")

    return {
        "mean_train": float(np.mean(train_scores)),
        "std_train": float(np.std(train_scores)),
        "mean_test": float(np.mean(test_scores)),
        "std_test": float(np.std(test_scores)),
        "runs": [
            {"seed": s, "train": tr, "test": te, "x": x}
            for s, tr, te, x in zip(seeds, train_scores, test_scores, best_configs)
        ]
    }


def main() -> None:
    datasets_to_run = [
        ("arithmetic", "data/arithmetic.jsonl"),
        ("logic", "data/logic.jsonl"),
    ]

    llm = OllamaLLM(LLMConfig())
    SEEDS = [0, 1, 2, 3, 4]

    final_results: Dict[str, Any] = {}

    for ds_name, ds_path in datasets_to_run:
        print(f"\n==========================================")
        print(f"Running on Dataset: {ds_name}")
        print(f"==========================================")

        if ds_name in ("toy_math", "arithmetic"):
            answer_type = "number"
            blocks_path = "prompts/blocks_number.json"
        elif ds_name in ("logic", "comparative"):
            answer_type = "yesno"
            blocks_path = "prompts/blocks_yesno.json"
        else:
            answer_type = "number"
            blocks_path = "prompts/blocks_number.json"

        if not Path(blocks_path).exists():
            print(f"Skipping {ds_name}: {blocks_path} not found.")
            continue

        if not Path(ds_path).exists():
            print(f"Skipping {ds_name}: {ds_path} not found.")
            continue

        blocks = load_blocks(blocks_path)
        print(f"Using blocks: {blocks_path} (n={len(blocks)}) | answer_type={answer_type}")

        full_data = load_dataset_jsonl(ds_path)
        train_data, test_data = split_train_test(full_data, train_ratio=0.8, seed=42)
        print(f"Data split: {len(train_data)} train, {len(test_data)} test")

        train_evaluator = PromptEvaluator(llm, train_data[:20], blocks, answer_type=answer_type)
        test_evaluator = PromptEvaluator(llm, test_data, blocks, answer_type=answer_type)

        ds_results: Dict[str, Any] = {}

        # --- Your deterministic baseline (bit-vector) ---
        print("Running Deterministic Baseline...")
        base_x, base_train_acc = run_baseline(train_evaluator)
        base_test_acc = test_evaluator.eval_accuracy(base_x)
        ds_results["baseline"] = {"train": base_train_acc, "test": base_test_acc, "x": base_x}
        print(f"  Train: {base_train_acc:.3f}, Test: {base_test_acc:.3f}")

        # --- DSPy MIPROv2 baseline (real) ---
        # Safe: if DSPy isn't installed, this will print an error but not kill the whole run.
        try:
            print("Running DSPy MIPROv2 Baseline...")
            dspy_train, dspy_test = run_dspy_miprov2_baseline(
                train_rows=train_data[:20],
                test_rows=test_data,
                answer_type=answer_type,
                auto="light",
                seed=0,
                base_url=LLMConfig().base_url,
            )
            ds_results["dspy_miprov2"] = {"train": float(dspy_train), "test": float(dspy_test)}
            print(f"  DSPy Train: {dspy_train:.3f}, DSPy Test: {dspy_test:.3f}")
        except Exception as e:
            print(f"  Skipping DSPy MIPROv2 (not available): {e}")

        # --- Metaheuristics ---
        ds_results["sa"] = run_multi_seed_algorithm(
            "Simulated Annealing",
            simulated_annealing,
            train_evaluator,
            test_evaluator,
            SEEDS,
            iters=20,
            t0=1.0,
            cooling=0.95,
        )

        ds_results["ga"] = run_multi_seed_algorithm(
            "Genetic Algorithm",
            genetic_algorithm,
            train_evaluator,
            test_evaluator,
            SEEDS,
            pop_size=8,
            generations=10,
            elitism=2,
        )

        ds_results["pso"] = run_multi_seed_algorithm(
            "PSO",
            binary_pso,
            train_evaluator,
            test_evaluator,
            SEEDS,
            swarm_size=8,
            iters=10,
            w_start=0.9,
            w_end=0.4,
        )

        final_results[ds_name] = ds_results

        print(f"\nSummary for {ds_name}:")
        print(f"  Baseline: {base_test_acc:.3f}")
        if "dspy_miprov2" in ds_results:
            print(f"  DSPy MIPROv2: {ds_results['dspy_miprov2']['test']:.3f}")
        print(f"  SA Mean:  {ds_results['sa']['mean_test']:.3f} ± {ds_results['sa']['std_test']:.3f}")
        print(f"  GA Mean:  {ds_results['ga']['mean_test']:.3f} ± {ds_results['ga']['std_test']:.3f}")
        print(f"  PSO Mean: {ds_results['pso']['mean_test']:.3f} ± {ds_results['pso']['std_test']:.3f}")

    save_json("results/final_scores.json", final_results)
    print("\nSaved results to results/final_scores.json")


if __name__ == "__main__":
    main()

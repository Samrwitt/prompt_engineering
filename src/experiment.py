from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from src.model import HFLLM
from src.fitness import load_dataset_jsonl, load_blocks, PromptEvaluator, split_train_test
from src.baseline import run_baseline, run_random_baseline, run_greedy_baseline
from src.sa import simulated_annealing
from src.ga import genetic_algorithm
from src.pso import binary_pso


def save_json(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    datasets_to_run = [
        ("toy_math", "data/toy_math.jsonl"),
        ("arithmetic", "data/arithmetic.jsonl"),
        ("logic", "data/logic.jsonl"),
    ]

    blocks = load_blocks("prompts/blocks.json")
    llm = HFLLM()  # set model_name in src/model.py if changing models

    final_results: Dict[str, Any] = {}

    for ds_name, ds_path in datasets_to_run:
        print(f"\n==========================================")
        print(f"Running on Dataset: {ds_name}")
        print(f"==========================================")

        full_data = load_dataset_jsonl(ds_path)

        train_data, test_data = split_train_test(full_data, train_ratio=0.8, seed=42)
        print(f"Data split: {len(train_data)} train, {len(test_data)} test")

        # Pick answer_type based on dataset
        if ds_name in ("toy_math", "arithmetic"):
            answer_type = "number"
        elif ds_name == "logic":
            answer_type = "yesno"  # change to "abcd" if needed
        else:
            answer_type = "number"

        train_evaluator = PromptEvaluator(llm, train_data, blocks, answer_type=answer_type)
        test_evaluator = PromptEvaluator(llm, test_data, blocks, answer_type=answer_type)

        ds_results: Dict[str, Any] = {}

        # --- Baselines ---
        print("Running Deterministic Baseline...")
        base_x, base_train_acc = run_baseline(train_evaluator)
        base_test_acc = test_evaluator.eval_accuracy(base_x)
        ds_results["baseline"] = {"train": base_train_acc, "test": base_test_acc, "x": base_x}
        print(f"  Train: {base_train_acc:.2f}, Test: {base_test_acc:.2f}")

        print("Running Random Baseline...")
        rand_x, rand_train_acc = run_random_baseline(train_evaluator, num_samples=5, seed=0)
        rand_test_acc = test_evaluator.eval_accuracy(rand_x)
        ds_results["random"] = {"train": rand_train_acc, "test": rand_test_acc, "x": rand_x}
        print(f"  Train: {rand_train_acc:.2f}, Test: {rand_test_acc:.2f}")

        print("Running Greedy Baseline...")
        greedy_x, greedy_train_acc = run_greedy_baseline(train_evaluator, steps=5, seed=0)
        greedy_test_acc = test_evaluator.eval_accuracy(greedy_x)
        ds_results["greedy"] = {"train": greedy_train_acc, "test": greedy_test_acc, "x": greedy_x}
        print(f"  Train: {greedy_train_acc:.2f}, Test: {greedy_test_acc:.2f}")

        # --- Metaheuristics (reduced params for quick verification) ---
        print("Running SA...")
        sa_x, sa_train_acc, _ = simulated_annealing(
            train_evaluator.eval_accuracy,
            n_dim=len(blocks),
            iters=5,
            seed=0,
        )
        sa_test_acc = test_evaluator.eval_accuracy(sa_x)
        ds_results["sa"] = {"train": sa_train_acc, "test": sa_test_acc, "x": sa_x}
        print(f"  Train: {sa_train_acc:.2f}, Test: {sa_test_acc:.2f}")

        print("Running GA...")
        ga_x, ga_train_acc, _ = genetic_algorithm(
            train_evaluator.eval_accuracy,
            n_dim=len(blocks),
            pop_size=4,
            generations=3,
            seed=0,
        )
        ga_test_acc = test_evaluator.eval_accuracy(ga_x)
        ds_results["ga"] = {"train": ga_train_acc, "test": ga_test_acc, "x": ga_x}
        print(f"  Train: {ga_train_acc:.2f}, Test: {ga_test_acc:.2f}")

        print("Running PSO...")
        pso_x, pso_train_acc, _ = binary_pso(
            train_evaluator.eval_accuracy,
            n_dim=len(blocks),
            swarm_size=4,
            iters=3,
            seed=0,
        )
        pso_test_acc = test_evaluator.eval_accuracy(pso_x)
        ds_results["pso"] = {"train": pso_train_acc, "test": pso_test_acc, "x": pso_x}
        print(f"  Train: {pso_train_acc:.2f}, Test: {pso_test_acc:.2f}")

        final_results[ds_name] = ds_results

    save_json("results/final_scores.json", final_results)
    print("\nSaved all results to results/final_scores.json")


if __name__ == "__main__":
    main()

"""
Main Experiment Runner

This module orchestrates the complete prompt optimization experiment, running
multiple optimization algorithms (baselines, metaheuristics, and DSPy SOTA)
across different datasets and generating comprehensive results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.model import HFLLM, LLMConfig
from src.fitness import load_dataset_jsonl, load_blocks, PromptEvaluator, split_train_test
from src.baseline import run_baseline, run_random_baseline, run_greedy_baseline
from src.sa import simulated_annealing
from src.ga import genetic_algorithm
from src.pso import binary_pso
from src.block_analysis import analyze_individual_blocks, analyze_solution_blocks
from src.sota_dspy import run_dspy_sota


def save_json(path: str, obj: Dict[str, Any]) -> None:
    """Save a dictionary to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def print_section(title: str, width: int = 60) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def main(
    run_block_analysis: bool = True,
    run_dspy: bool = True,
    save_curves: bool = True,
) -> None:
    """
    Main experiment runner.
    
    Args:
        run_block_analysis: Whether to run block effectiveness analysis
        run_dspy: Whether to run DSPy SOTA baseline
        save_curves: Whether to save convergence curves
    """
    print_section("Prompt Optimization Experiment", width=70)
    print("Comparing metaheuristic algorithms for prompt block selection")
    print("=" * 70)
    
    datasets_to_run = [
        ("arithmetic", "data/arithmetic.jsonl"),
        ("logic", "data/logic.jsonl"),
        ("comparative", "data/comparative_reasoning.jsonl"),
    ]

    # Initialize models
    print("\nInitializing models...")
    try:
        llm_yesno = HFLLM(LLMConfig(
            model_name="google/flan-t5-small",
            device="cpu",
            max_new_tokens=16
        ))
        print("  ✓ FLAN-T5-small loaded (for yes/no tasks)")
    except Exception as e:
        print(f"  ✗ Error loading FLAN-T5: {e}")
        sys.exit(1)
    
    try:
        llm_math = HFLLM(LLMConfig(
            model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
            device="cpu",
            max_new_tokens=32
        ))
        print("  ✓ Qwen2.5-Math loaded (for arithmetic tasks)")
    except Exception as e:
        print(f"  ✗ Error loading Qwen2.5-Math: {e}")
        print("  ⚠ Falling back to FLAN-T5 for arithmetic")
        llm_math = llm_yesno

    final_results: Dict[str, Any] = {}
    all_curves: Dict[str, Dict[str, List[float]]] = {}

    for ds_name, ds_path in datasets_to_run:
        print_section(f"Dataset: {ds_name.upper()}", width=70)

        # 1) Pick answer_type + blocks_path based on dataset
        if ds_name in ("toy_math", "arithmetic"):
            answer_type = "number"
            blocks_path = "prompts/blocks_number.json"
        elif ds_name in ("logic", "comparative"):
            answer_type = "yesno"
            blocks_path = "prompts/blocks_yesno.json"
        else:
            answer_type = "number"
            blocks_path = "prompts/blocks_number.json"

        # 2) Load blocks PER dataset type
        try:
            blocks = load_blocks(blocks_path)
            print(f"Loaded {len(blocks)} prompt blocks from {blocks_path}")
            print(f"Answer type: {answer_type}")
        except Exception as e:
            print(f"Error loading blocks: {e}")
            continue

        # 3) Load and split dataset
        try:
            full_data = load_dataset_jsonl(ds_path)
            train_data, test_data = split_train_test(full_data, train_ratio=0.8, seed=42)
            print(f"Dataset: {len(full_data)} total | {len(train_data)} train | {len(test_data)} test")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            continue

        # 4) Select appropriate model
        if ds_name in ("toy_math", "arithmetic"):
            llm = llm_math
        else:
            llm = llm_yesno

        train_evaluator = PromptEvaluator(llm, train_data, blocks, answer_type=answer_type)
        test_evaluator = PromptEvaluator(llm, test_data, blocks, answer_type=answer_type)

        # Block analysis (optional)
        if run_block_analysis:
            print_section("Block Effectiveness Analysis", width=70)
            try:
                block_analysis = analyze_individual_blocks(train_evaluator, verbose=True)
                ds_results: Dict[str, Any] = {"block_analysis": block_analysis}
            except Exception as e:
                print(f"Error in block analysis: {e}")
                ds_results: Dict[str, Any] = {}
        else:
            ds_results: Dict[str, Any] = {}

        print_section("Running Optimization Algorithms", width=70)

        # --- Baselines ---
        print("\n[1/6] Deterministic Baseline (DSPy-inspired)...")
        try:
            base_x, base_train_acc = run_baseline(train_evaluator)
            base_test_acc = test_evaluator.eval_accuracy(base_x)
            ds_results["baseline"] = {
                "train": base_train_acc,
                "test": base_test_acc,
                "x": base_x,
                "blocks_path": blocks_path,
                "answer_type": answer_type,
            }
            print(f"  ✓ Train: {base_train_acc:.4f} | Test: {base_test_acc:.4f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            ds_results["baseline"] = {"train": 0.0, "test": 0.0, "x": []}

        print("\n[2/6] Random Baseline...")
        try:
            rand_x, rand_train_acc = run_random_baseline(train_evaluator, num_samples=20, seed=0)
            rand_test_acc = test_evaluator.eval_accuracy(rand_x)
            ds_results["random"] = {"train": rand_train_acc, "test": rand_test_acc, "x": rand_x}
            print(f"  ✓ Train: {rand_train_acc:.4f} | Test: {rand_test_acc:.4f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            ds_results["random"] = {"train": 0.0, "test": 0.0, "x": []}

        print("\n[3/6] Greedy Baseline...")
        try:
            greedy_x, greedy_train_acc = run_greedy_baseline(
                train_evaluator, steps=min(20, len(blocks) * 2), seed=0
            )
            greedy_test_acc = test_evaluator.eval_accuracy(greedy_x)
            ds_results["greedy"] = {"train": greedy_train_acc, "test": greedy_test_acc, "x": greedy_x}
            print(f"  ✓ Train: {greedy_train_acc:.4f} | Test: {greedy_test_acc:.4f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            ds_results["greedy"] = {"train": 0.0, "test": 0.0, "x": []}

        # --- Metaheuristics ---
        print("\n[4/6] Simulated Annealing (SA)...")
        try:
            sa_x, sa_train_acc, sa_curve = simulated_annealing(
                train_evaluator.eval_accuracy,
                n_dim=len(blocks),
                iters=50,
                seed=0,
            )
            sa_test_acc = test_evaluator.eval_accuracy(sa_x)
            ds_results["sa"] = {"train": sa_train_acc, "test": sa_test_acc, "x": sa_x}
            if save_curves:
                if ds_name not in all_curves:
                    all_curves[ds_name] = {}
                all_curves[ds_name]["sa"] = sa_curve
            print(f"  ✓ Train: {sa_train_acc:.4f} | Test: {sa_test_acc:.4f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            ds_results["sa"] = {"train": 0.0, "test": 0.0, "x": []}

        print("\n[5/6] Genetic Algorithm (GA)...")
        try:
            ga_x, ga_train_acc, ga_curve = genetic_algorithm(
                train_evaluator.eval_accuracy,
                n_dim=len(blocks),
                pop_size=16,
                generations=20,
                seed=0,
            )
            ga_test_acc = test_evaluator.eval_accuracy(ga_x)
            ds_results["ga"] = {"train": ga_train_acc, "test": ga_test_acc, "x": ga_x}
            if save_curves:
                if ds_name not in all_curves:
                    all_curves[ds_name] = {}
                all_curves[ds_name]["ga"] = ga_curve
            print(f"  ✓ Train: {ga_train_acc:.4f} | Test: {ga_test_acc:.4f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            ds_results["ga"] = {"train": 0.0, "test": 0.0, "x": []}

        print("\n[6/6] Particle Swarm Optimization (PSO)...")
        try:
            pso_x, pso_train_acc, pso_curve = binary_pso(
                train_evaluator.eval_accuracy,
                n_dim=len(blocks),
                swarm_size=16,
                iters=20,
                seed=0,
            )
            pso_test_acc = test_evaluator.eval_accuracy(pso_x)
            ds_results["pso"] = {"train": pso_train_acc, "test": pso_test_acc, "x": pso_x}
            if save_curves:
                if ds_name not in all_curves:
                    all_curves[ds_name] = {}
                all_curves[ds_name]["pso"] = pso_curve
            print(f"  ✓ Train: {pso_train_acc:.4f} | Test: {pso_test_acc:.4f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            ds_results["pso"] = {"train": 0.0, "test": 0.0, "x": []}

        # --- DSPy SOTA Baseline ---
        if run_dspy:
            print("\n[7/7] DSPy SOTA (COPRO)...")
            try:
                dspy_train_acc, dspy_test_acc = run_dspy_sota(
                    llm, train_data, test_data, answer_type=answer_type, verbose=True
                )
                ds_results["dspy_sota"] = {
                    "train": dspy_train_acc,
                    "test": dspy_test_acc,
                }
                print(f"  ✓ Train: {dspy_train_acc:.4f} | Test: {dspy_test_acc:.4f}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                ds_results["dspy_sota"] = {"train": 0.0, "test": 0.0}

        # Analyze best solution
        print_section("Best Solution Analysis", width=70)
        best_method = max(
            [k for k in ds_results.keys() if k not in ("block_analysis", "dspy_sota")],
            key=lambda k: ds_results[k].get("test", 0.0)
        )
        best_x = ds_results[best_method].get("x", [])
        if best_x:
            try:
                analyze_solution_blocks(best_x, train_evaluator, verbose=True)
                ds_results["best_method"] = best_method
                ds_results["best_test_acc"] = ds_results[best_method].get("test", 0.0)
            except Exception as e:
                print(f"Error analyzing best solution: {e}")

        final_results[ds_name] = ds_results

    # Save results
    print_section("Saving Results", width=70)
    save_json("results/final_scores.json", final_results)
    print("  ✓ Saved results/final_scores.json")
    
    if save_curves and all_curves:
        save_json("results/curves.json", all_curves)
        print("  ✓ Saved results/curves.json")
    
    # Print summary
    print_section("Experiment Summary", width=70)
    for ds_name, results in final_results.items():
        print(f"\n{ds_name.upper()}:")
        for method, data in results.items():
            if isinstance(data, dict) and "test" in data:
                print(f"  {method:15s}: {data.get('test', 0.0):.4f} (test) | {data.get('train', 0.0):.4f} (train)")
    
    print("\n" + "=" * 70)
    print("Experiment completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main(run_block_analysis=True, run_dspy=True, save_curves=True)

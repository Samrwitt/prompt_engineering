from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm
from scipy.stats import ranksums
import sys
# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import HFLLM
from src.fitness import load_dataset_jsonl, load_blocks, PromptEvaluator, extract_number
from src.baseline import run_baseline, run_random_baseline, run_greedy_baseline
from src.sa import simulated_annealing
from src.ga import genetic_algorithm
from src.pso import binary_pso
# from src.sota_dspy import run_dspy_sota, load_dataset_jsonl as load_dspy_data



def save_json(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    # 1. Define datasets
    datasets_to_run = [
        ("toy_math", "data/toy_math.jsonl"),
        ("arithmetic", "data/arithmetic.jsonl"),
        ("logic", "data/logic.jsonl")
    ]
    
    blocks = load_blocks("prompts/blocks.json")
    llm = HFLLM()  # flan-t5-small on CPU

    final_results = {}

    for ds_name, ds_path in datasets_to_run:
        print(f"\n==========================================")
        print(f"Running on Dataset: {ds_name}")
        print(f"==========================================")
        
        full_data = load_dataset_jsonl(ds_path)
        # 2. Split Train/Test
        from src.fitness import split_train_test
        train_data, test_data = split_train_test(full_data, train_ratio=0.8, seed=42)
        print(f"Data split: {len(train_data)} train, {len(test_data)} test")
        
        train_evaluator = PromptEvaluator(llm, train_data, blocks)
        test_evaluator = PromptEvaluator(llm, test_data, blocks)
        
        ds_results = {}
        
        # --- Baselines ---
        
        # Deterministic
        print("Running Deterministic Baseline...")
        base_x, base_train_acc = run_baseline(train_evaluator)
        base_test_acc = test_evaluator.eval_accuracy(base_x)
        ds_results["baseline"] = {"train": base_train_acc, "test": base_test_acc}
        print(f"  Train: {base_train_acc:.2f}, Test: {base_test_acc:.2f}")

        # Random Search - reduced samples for speed
        print("Running Random Baseline...")
        rand_x, rand_train_acc = run_random_baseline(train_evaluator, num_samples=5, seed=0)
        rand_test_acc = test_evaluator.eval_accuracy(rand_x)
        ds_results["random"] = {"train": rand_train_acc, "test": rand_test_acc}
        print(f"  Train: {rand_train_acc:.2f}, Test: {rand_test_acc:.2f}")

        # Greedy - reduced steps for speed
        print("Running Greedy Baseline...")
        greedy_x, greedy_train_acc = run_greedy_baseline(train_evaluator, steps=5, seed=0)
        greedy_test_acc = test_evaluator.eval_accuracy(greedy_x)
        ds_results["greedy"] = {"train": greedy_train_acc, "test": greedy_test_acc}
        print(f"  Train: {greedy_train_acc:.2f}, Test: {greedy_test_acc:.2f}")

        # --- Metaheuristics (Single run for simplicity in this loop) ---
        # Reduced iterations/params for speed verification
        
        # SA
        print("Running SA...")
        sa_x, sa_train_acc, _ = simulated_annealing(
            train_evaluator.eval_accuracy, 
            n_dim=len(blocks), 
            iters=5,  # Reduced
            seed=0
        )
        sa_test_acc = test_evaluator.eval_accuracy(sa_x)
        ds_results["sa"] = {"train": sa_train_acc, "test": sa_test_acc}
        print(f"  Train: {sa_train_acc:.2f}, Test: {sa_test_acc:.2f}")

        # GA
        print("Running GA...")
        ga_x, ga_train_acc, _ = genetic_algorithm(
            train_evaluator.eval_accuracy, 
            n_dim=len(blocks), 
            pop_size=4,   # Reduced
            generations=3, # Reduced
            seed=0
        )
        ga_test_acc = test_evaluator.eval_accuracy(ga_x)
        ds_results["ga"] = {"train": ga_train_acc, "test": ga_test_acc}
        print(f"  Train: {ga_train_acc:.2f}, Test: {ga_test_acc:.2f}")

        # PSO
        print("Running PSO...")
        pso_x, pso_train_acc, _ = binary_pso(
            train_evaluator.eval_accuracy, 
            n_dim=len(blocks), 
            swarm_size=4, # Reduced
            iters=3,      # Reduced
            seed=0
        )
        pso_test_acc = test_evaluator.eval_accuracy(pso_x)
        ds_results["pso"] = {"train": pso_train_acc, "test": pso_test_acc}
        print(f"  Train: {pso_train_acc:.2f}, Test: {pso_test_acc:.2f}")
        
        final_results[ds_name] = ds_results
        
    # Save all results
    save_json("results/final_scores.json", final_results)
    print("\nSaved all results to results/final_scores.json")


if __name__ == "__main__":
    main()


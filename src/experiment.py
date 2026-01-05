from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm
from scipy.stats import ranksums

from src.model import HFLLM
from src.fitness import load_dataset_jsonl, load_blocks, PromptEvaluator
from src.baseline import run_baseline
from src.sa import simulated_annealing
from src.ga import genetic_algorithm
from src.pso import binary_pso


def save_json(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    dataset = load_dataset_jsonl("data/toy_math.jsonl")
    blocks = load_blocks("prompts/blocks.json")

    llm = HFLLM()  # flan-t5-small on CPU
    evaluator = PromptEvaluator(llm, dataset, blocks)

    # Baseline
    base_acc = run_baseline(evaluator)
    print("Baseline accuracy:", base_acc)

    # Heuristics (single runs for now; later do multiple seeds)
    sa_x, sa_f, sa_curve = simulated_annealing(evaluator.eval_accuracy, n_dim=len(blocks), seed=0)
    ga_x, ga_f, ga_curve = genetic_algorithm(evaluator.eval_accuracy, n_dim=len(blocks), seed=0)
    pso_x, pso_f, pso_curve = binary_pso(evaluator.eval_accuracy, n_dim=len(blocks), seed=0)

    print("SA best:", sa_f, sa_x)
    print("GA best:", ga_f, ga_x)
    print("PSO best:", pso_f, pso_x)

    # Save curves (for plots later)
    save_json("results/curves.json", {
        "baseline_accuracy": base_acc,
        "sa_curve": sa_curve,
        "ga_curve": ga_curve,
        "pso_curve": pso_curve,
        "sa_best": {"x": sa_x, "fitness": sa_f},
        "ga_best": {"x": ga_x, "fitness": ga_f},
        "pso_best": {"x": pso_x, "fitness": pso_f},
        "blocks": blocks,
    })

    # Quick stats demo with multiple seeds (small)
    seeds = [0, 1, 2, 3, 4]
    sa_scores, ga_scores, pso_scores = [], [], []

    for s in tqdm(seeds, desc="Multi-seed eval"):
        _, f, _ = simulated_annealing(evaluator.eval_accuracy, n_dim=len(blocks), seed=s)
        sa_scores.append(f)
        _, f, _ = genetic_algorithm(evaluator.eval_accuracy, n_dim=len(blocks), seed=s)
        ga_scores.append(f)
        _, f, _ = binary_pso(evaluator.eval_accuracy, n_dim=len(blocks), seed=s)
        pso_scores.append(f)

    # ---- DEBUG: show one example end-to-end ----
    x_debug = [1] * len(blocks)  # turn on all blocks
    print("\n--- DEBUG ONE SAMPLE ---")
    print("Blocks ON:", x_debug)

    item0 = dataset[0]
    print("Dataset sample keys:", item0.keys())
    print("Q:", repr(item0.get("q")))
    print("A:", repr(item0.get("a")))

    prompt_prefix = "\n".join(blocks)
    inp = f"{prompt_prefix}\n\nQuestion: {item0['q']}\nAnswer:"
    print("\nINPUT TO MODEL:\n", inp)

    out = llm.generate(inp)
    print("\nRAW MODEL OUTPUT:", repr(out))

    from src.fitness import extract_number
    pred = extract_number(out)
    print("EXTRACTED:", repr(pred), "GT:", repr(item0['a'].strip()))
    print("--- END DEBUG ---\n")
    # -------------------------------------------


    # Wilcoxon rank-sum (ranksums) pairwise example
    # (You will report these properly in the paper)
    print("Ranksums SA vs GA:", ranksums(sa_scores, ga_scores))
    print("Ranksums SA vs PSO:", ranksums(sa_scores, pso_scores))
    print("Ranksums GA vs PSO:", ranksums(ga_scores, pso_scores))

    save_json("results/final_scores.json", {
        "seeds": seeds,
        "sa_scores": sa_scores,
        "ga_scores": ga_scores,
        "pso_scores": pso_scores,
        "baseline_accuracy": base_acc
    })


if __name__ == "__main__":
    main()

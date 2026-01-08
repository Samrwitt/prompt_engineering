"""
Block Effectiveness Analysis Module

This module provides functionality to analyze which prompt blocks are most effective
by evaluating individual blocks and combinations.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple
from src.fitness import PromptEvaluator, build_prompt


def analyze_individual_blocks(
    evaluator: PromptEvaluator,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Analyze the effectiveness of each individual prompt block.
    
    Args:
        evaluator: PromptEvaluator instance with dataset and blocks
        verbose: Whether to print results
        
    Returns:
        Dictionary with block analysis results
    """
    n_blocks = len(evaluator.blocks)
    block_scores: Dict[int, float] = {}
    
    if verbose:
        print("\n" + "=" * 60)
        print("Individual Block Analysis")
        print("=" * 60)
    
    # Test each block individually
    for i in range(n_blocks):
        x = [0] * n_blocks
        x[i] = 1
        acc = evaluator.eval_accuracy(x)
        block_scores[i] = acc
        
        if verbose:
            print(f"Block {i+1}: {acc:.4f} | {evaluator.blocks[i][:60]}...")
    
    # Sort by effectiveness
    sorted_blocks = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)
    
    result = {
        "individual_scores": block_scores,
        "sorted_by_effectiveness": [
            {
                "index": idx,
                "block": evaluator.blocks[idx],
                "accuracy": score,
            }
            for idx, score in sorted_blocks
        ],
        "best_block": {
            "index": sorted_blocks[0][0],
            "block": evaluator.blocks[sorted_blocks[0][0]],
            "accuracy": sorted_blocks[0][1],
        },
    }
    
    if verbose:
        print("\nTop 3 Most Effective Blocks:")
        for rank, (idx, score) in enumerate(sorted_blocks[:3], 1):
            print(f"  {rank}. Block {idx+1} ({score:.4f}): {evaluator.blocks[idx]}")
    
    return result


def analyze_block_combinations(
    evaluator: PromptEvaluator,
    top_k: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Analyze effectiveness of combinations of top-k blocks.
    
    Args:
        evaluator: PromptEvaluator instance
        top_k: Number of top blocks to combine
        verbose: Whether to print results
        
    Returns:
        Dictionary with combination analysis results
    """
    # First get individual block scores
    individual_analysis = analyze_individual_blocks(evaluator, verbose=False)
    sorted_blocks = individual_analysis["sorted_by_effectiveness"][:top_k]
    
    n_blocks = len(evaluator.blocks)
    combinations: List[Dict[str, Any]] = []
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Top-{top_k} Block Combinations Analysis")
        print("=" * 60)
    
    # Test all combinations of top-k blocks
    from itertools import combinations
    
    for r in range(1, min(top_k + 1, len(sorted_blocks) + 1)):
        for combo in combinations(range(len(sorted_blocks)), r):
            x = [0] * n_blocks
            for idx in combo:
                block_idx = sorted_blocks[idx]["index"]
                x[block_idx] = 1
            
            acc = evaluator.eval_accuracy(x)
            combo_indices = [sorted_blocks[idx]["index"] for idx in combo]
            combo_text = [sorted_blocks[idx]["block"] for idx in combo]
            
            combinations.append({
                "indices": combo_indices,
                "blocks": combo_text,
                "accuracy": acc,
                "size": len(combo),
            })
            
            if verbose:
                print(f"  Size {len(combo)}: {acc:.4f} | Blocks: {combo_indices}")
    
    # Sort by accuracy
    combinations.sort(key=lambda x: x["accuracy"], reverse=True)
    
    result = {
        "all_combinations": combinations,
        "best_combination": combinations[0] if combinations else None,
    }
    
    if verbose and combinations:
        print(f"\nBest Combination (accuracy: {combinations[0]['accuracy']:.4f}):")
        for i, block in enumerate(combinations[0]["blocks"], 1):
            print(f"  {i}. {block}")
    
    return result


def analyze_solution_blocks(
    solution_x: List[int],
    evaluator: PromptEvaluator,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Analyze which blocks are used in a given solution.
    
    Args:
        solution_x: Binary vector representing selected blocks
        evaluator: PromptEvaluator instance
        verbose: Whether to print results
        
    Returns:
        Dictionary with solution analysis
    """
    selected_blocks = [
        (i, evaluator.blocks[i])
        for i, bit in enumerate(solution_x)
        if bit == 1
    ]
    
    prompt = build_prompt(evaluator.blocks, solution_x)
    
    result = {
        "selected_indices": [i for i, bit in enumerate(solution_x) if bit == 1],
        "selected_blocks": [block for _, block in selected_blocks],
        "num_blocks": len(selected_blocks),
        "prompt": prompt,
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Solution Block Analysis")
        print("=" * 60)
        print(f"Number of blocks selected: {len(selected_blocks)}")
        print("\nSelected blocks:")
        for i, (idx, block) in enumerate(selected_blocks, 1):
            print(f"  {i}. Block {idx+1}: {block}")
        print(f"\nFull prompt:\n{prompt}")
    
    return result


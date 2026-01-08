from __future__ import annotations

import math
import random
from typing import List, Tuple, Dict
from tqdm import tqdm


def neighbor_flip(x: List[int], k: int = 1) -> List[int]:
    y = x[:]
    idxs = random.sample(range(len(x)), k=min(k, len(x)))
    for i in idxs:
        y[i] = 1 - y[i]
    return y


def simulated_annealing(
    eval_fn,
    n_dim: int,
    iters: int = 100,  # Increased default for research level
    t0: float = 1.0,
    cooling: float = 0.95,
    flips_per_move: int = 1,
    seed: int = 0,
) -> Tuple[List[int], float, List[float]]:
    """
    Simulated Annealing with Geometric Cooling.
    
    Research Standards:
    - Geometric cooling schedule: T_k = T_0 * (cooling)^k
    - Acceptance probability: P = exp((f_new - f_old) / T)
    - Returns curve for convergence analysis.
    """
    random.seed(seed)

    x = [random.randint(0, 1) for _ in range(n_dim)]
    fx = eval_fn(x)

    best_x, best_f = x[:], fx
    curve = [best_f]

    T = t0
    # Wrap iterations with tqdm for visibility
    for _ in tqdm(range(iters), desc="SA Optimization", leave=False):
        cand = neighbor_flip(x, k=flips_per_move)
        f_cand = eval_fn(cand)

        # Metropolis Criterion
        # If better, always accept (delta > 0 for maximization)
        if f_cand >= fx:
            x, fx = cand, f_cand
        else:
            # If worse, accept with prob exp(delta / T)
            # delta is negative here (f_cand - fx)
            delta = f_cand - fx
            p = math.exp(delta / max(T, 1e-9))
            if random.random() < p:
                x, fx = cand, f_cand

        # Keep track of global best found so far
        if fx > best_f:
            best_x, best_f = x[:], fx

        # Geometric Cooling
        T *= cooling
        curve.append(best_f)

    return best_x, best_f, curve

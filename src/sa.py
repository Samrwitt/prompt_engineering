from __future__ import annotations

import math
import random
from typing import List, Tuple, Dict


def neighbor_flip(x: List[int], k: int = 1) -> List[int]:
    y = x[:]
    idxs = random.sample(range(len(x)), k=min(k, len(x)))
    for i in idxs:
        y[i] = 1 - y[i]
    return y


def simulated_annealing(
    eval_fn,
    n_dim: int,
    iters: int = 60,
    t0: float = 1.0,
    cooling: float = 0.97,
    flips_per_move: int = 1,
    seed: int = 0,
) -> Tuple[List[int], float, List[float]]:
    random.seed(seed)

    x = [random.randint(0, 1) for _ in range(n_dim)]
    fx = eval_fn(x)

    best_x, best_f = x[:], fx
    curve = [best_f]

    T = t0
    for _ in range(iters):
        cand = neighbor_flip(x, k=flips_per_move)
        f_cand = eval_fn(cand)

        if f_cand >= fx:
            x, fx = cand, f_cand
        else:
            # accept worse with prob exp((f_new - f_old)/T)
            p = math.exp((f_cand - fx) / max(T, 1e-9))
            if random.random() < p:
                x, fx = cand, f_cand

        if fx > best_f:
            best_x, best_f = x[:], fx

        T *= cooling
        curve.append(best_f)

    return best_x, best_f, curve

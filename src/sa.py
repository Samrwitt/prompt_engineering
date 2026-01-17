from __future__ import annotations

import math
import random
from typing import List, Tuple, Dict
from tqdm import tqdm

def _flip_k(x: List[int], rng: random.Random, k: int) -> List[int]:
    y = x[:]
    idxs = rng.sample(range(len(x)), k=min(k, len(x)))
    for i in idxs:
        y[i] = 1 - y[i]
    return y

def simulated_annealing_sharp(
    eval_fn,
    n_dim: int,
    iters: int = 60,
    t0: float = 1.0,
    cooling: float = 0.97,
    seed: int = 0,
    stagnation_reheat: int = 12,
    patience: int = 20,          # early stop if no best improvement
    tabu_size: int = 200,        # avoid re-evaluating same x
) -> Tuple[List[int], float, List[float]]:
    """
    SA++ for expensive eval_fn:
      - tabu cache to avoid duplicate evals
      - adaptive neighborhood: larger k early, smaller k late
      - reheating when stuck
      - early stop on stagnation of BEST
    """
    rng = random.Random(seed)

    def eval_cached(x: List[int], cache: Dict[Tuple[int, ...], float]) -> float:
        key = tuple(x)
        if key in cache:
            return cache[key]
        v = float(eval_fn(x))
        cache[key] = v
        # soft cap cache (keep it bounded)
        if len(cache) > tabu_size * 2:
            # random eviction (cheap, good enough)
            for _ in range(len(cache) - tabu_size):
                cache.pop(next(iter(cache)))
        return v

    x = [rng.randint(0, 1) for _ in range(n_dim)]
    cache: Dict[Tuple[int, ...], float] = {}
    fx = eval_cached(x, cache)

    best_x, best_f = x[:], fx
    curve = [best_f]

    T = float(t0)
    no_improve = 0
    no_best_improve = 0

    for step in tqdm(range(iters), desc="SA++", leave=False):
        # adaptive k: explore early, exploit late
        frac = step / max(1, iters - 1)
        if frac < 0.33:
            k = 2 if rng.random() < 0.7 else 3
        elif frac < 0.8:
            k = 1 if rng.random() < 0.75 else 2
        else:
            k = 1  # sharp exploitation at the end

        cand = _flip_k(x, rng, k)
        f_cand = eval_cached(cand, cache)

        if f_cand >= fx:
            x, fx = cand, f_cand
        else:
            delta = f_cand - fx
            p = math.exp(delta / max(T, 1e-9))
            if rng.random() < p:
                x, fx = cand, f_cand

        if fx > best_f:
            best_x, best_f = x[:], fx
            no_improve = 0
            no_best_improve = 0
        else:
            no_improve += 1
            no_best_improve += 1

        T *= cooling

        # reheat if stuck locally
        if no_improve >= stagnation_reheat:
            T = max(T, t0 * 0.6)  # sharper than 0.5 in practice
            no_improve = 0

        curve.append(best_f)

        # early stop if BEST isn't improving
        if no_best_improve >= patience:
            break

    return best_x, best_f, curve

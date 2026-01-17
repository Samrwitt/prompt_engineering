from __future__ import annotations

import math
import random
from typing import List, Tuple
from tqdm import tqdm

def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def _to_binary_temp(z: List[float], tau: float = 0.5, T: float = 1.0) -> List[int]:
    T = max(T, 1e-6)
    return [1 if _sigmoid(v / T) > tau else 0 for v in z]

def _clip_vec(u: List[float], lo: float = -6.0, hi: float = 6.0) -> List[float]:
    return [min(hi, max(lo, v)) for v in u]

def differential_evolution_binary_sharp(
    eval_fn,
    n_dim: int,
    pop_size: int = 18,
    iters: int = 25,
    F: float = 0.7,
    CR: float = 0.8,
    seed: int = 0,
    # sharpening knobs
    T0: float = 1.5,
    Tend: float = 0.35,
    tau: float = 0.5,
    jitter: float = 0.01,
) -> Tuple[List[int], float, List[float]]:
    """
    Sharper DE for binary prompt-block selection:
    - DE/current-to-best/1/bin (more exploitative)
    - temperature-annealed binarization (decisive late)
    - bounded logits to prevent saturation
    Returns (best_x, best_f, curve)
    """
    rng = random.Random(seed)

    Z = [[rng.uniform(-1.0, 1.0) for _ in range(n_dim)] for _ in range(pop_size)]

    def temp_at(t: int) -> float:
        # geometric anneal
        if iters <= 1:
            return Tend
        alpha = t / (iters - 1)
        return T0 * ((Tend / T0) ** alpha)

    # init fitness
    T = temp_at(0)
    X = [_to_binary_temp(z, tau=tau, T=T) for z in Z]
    fits = [eval_fn(x) for x in X]

    best_i = max(range(pop_size), key=lambda i: fits[i])
    best_x, best_f = X[best_i][:], float(fits[best_i])
    curve = [best_f]

    for t in tqdm(range(iters), desc="DE(sharp)", leave=False):
        T = temp_at(t)

        for i in range(pop_size):
            idxs = list(range(pop_size))
            idxs.remove(i)
            b, c = rng.sample(idxs, 2)

            # current-to-best/1 mutation (sharper exploitation)
            v = []
            for d in range(n_dim):
                val = (
                    Z[i][d]
                    + F * (Z[best_i][d] - Z[i][d])
                    + F * (Z[b][d] - Z[c][d])
                )
                # tiny jitter helps break ties in discrete mapping
                val += rng.uniform(-jitter, jitter)
                v.append(val)

            # binomial crossover
            j_rand = rng.randrange(n_dim)
            u = []
            for d in range(n_dim):
                if rng.random() < CR or d == j_rand:
                    u.append(v[d])
                else:
                    u.append(Z[i][d])

            u = _clip_vec(u)

            x_u = _to_binary_temp(u, tau=tau, T=T)
            f_u = eval_fn(x_u)

            if f_u >= fits[i]:
                Z[i] = u
                X[i] = x_u
                fits[i] = f_u
                if f_u > best_f:
                    best_f = float(f_u)
                    best_x = x_u[:]
                    best_i = i  # keep best index synced

        curve.append(best_f)

    return best_x, best_f, curve

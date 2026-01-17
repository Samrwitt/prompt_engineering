# src/hybrid.py
from __future__ import annotations
import math, random
from typing import List, Tuple, Callable
from tqdm import tqdm

def _sigmoid(z: float) -> float:
    # stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def _decode(z: List[float], tau: float = 0.5) -> List[int]:
    return [1 if _sigmoid(v) >= tau else 0 for v in z]

def _flip_k(x: List[int], rng: random.Random, k: int) -> List[int]:
    y = x[:]
    idxs = rng.sample(range(len(x)), k=min(k, len(x)))
    for i in idxs:
        y[i] = 1 - y[i]
    return y

def sa_sharp(
    eval_fn: Callable[[List[int]], float],
    x0: List[int],
    iters: int = 60,
    t0: float = 0.8,
    cooling: float = 0.965,
    seed: int = 0,
    reheat_every: int = 14,
) -> Tuple[List[int], float]:
    rng = random.Random(seed)
    x = x0[:]
    fx = float(eval_fn(x))
    best_x, best_f = x[:], fx
    T = float(t0)
    stale = 0

    for step in range(iters):
        # 1-flip most of the time, sometimes 2–3
        r = rng.random()
        k = 1 if r < 0.78 else (2 if r < 0.93 else 3)

        cand = _flip_k(x, rng, k)
        f_cand = float(eval_fn(cand))

        if f_cand >= fx:
            x, fx = cand, f_cand
        else:
            delta = f_cand - fx
            if rng.random() < math.exp(delta / max(T, 1e-9)):
                x, fx = cand, f_cand

        if fx > best_f:
            best_x, best_f = x[:], fx
            stale = 0
        else:
            stale += 1

        T *= cooling
        if stale >= reheat_every:
            T = max(T, t0 * 0.6)  # controlled reheat
            stale = 0

    return best_x, best_f

def de_sharp_binary(
    eval_fn: Callable[[List[int]], float],
    n_dim: int,
    pop_size: int = 18,
    iters: int = 25,
    tau: float = 0.5,
    seed: int = 0,
    p_best: float = 0.3,      # best-of-top-p mutation bias
    jDE_tau1: float = 0.1,    # F adapt prob
    jDE_tau2: float = 0.1,    # CR adapt prob
) -> Tuple[List[int], float, List[float]]:
    rng = random.Random(seed)

    # init logits
    Z = [[rng.uniform(-1.0, 1.0) for _ in range(n_dim)] for _ in range(pop_size)]
    X = [_decode(z, tau=tau) for z in Z]
    Fv = [rng.uniform(0.4, 0.9) for _ in range(pop_size)]
    CRv = [rng.uniform(0.5, 0.95) for _ in range(pop_size)]
    fits = [float(eval_fn(x)) for x in X]

    curve: List[float] = []
    for _ in tqdm(range(iters), desc="DE+", leave=False):
        # rank for p-best selection
        order = sorted(range(pop_size), key=lambda i: fits[i], reverse=True)
        topk = max(2, int(pop_size * p_best))

        for i in range(pop_size):
            # self-adapt F/CR
            if rng.random() < jDE_tau1:
                Fv[i] = rng.uniform(0.4, 0.95)
            if rng.random() < jDE_tau2:
                CRv[i] = rng.uniform(0.3, 0.99)

            F = Fv[i]
            CR = CRv[i]

            # choose pbest, r1, r2
            pbest = order[rng.randrange(topk)]
            idxs = list(range(pop_size))
            idxs.remove(i)
            if pbest in idxs:
                idxs.remove(pbest)
            r1, r2 = rng.sample(idxs, 2)

            # current-to-pbest/1
            v = [
                Z[i][d] + F * (Z[pbest][d] - Z[i][d]) + F * (Z[r1][d] - Z[r2][d])
                for d in range(n_dim)
            ]

            # binomial crossover
            j_rand = rng.randrange(n_dim)
            u = []
            for d in range(n_dim):
                u.append(v[d] if (rng.random() < CR or d == j_rand) else Z[i][d])

            x_u = _decode(u, tau=tau)
            f_u = float(eval_fn(x_u))

            if f_u >= fits[i]:
                Z[i] = u
                X[i] = x_u
                fits[i] = f_u

        curve.append(float(max(fits)))

    best_i = max(range(pop_size), key=lambda i: fits[i])
    return X[best_i][:], float(fits[best_i]), curve

def hybrid_de_sa(
    eval_fn: Callable[[List[int]], float],
    n_dim: int,
    seed: int = 0,
    # DE+
    pop_size: int = 18,
    de_iters: int = 25,
    # SA+
    sa_iters: int = 60,
) -> Tuple[List[int], float, List[float]]:
    best_x, best_f, curve = de_sharp_binary(
        eval_fn=eval_fn, n_dim=n_dim, pop_size=pop_size, iters=de_iters, seed=seed
    )
    rx, rf = sa_sharp(eval_fn=eval_fn, x0=best_x, iters=sa_iters, seed=seed)
    curve2 = curve + [float(rf)]
    return rx, float(rf), curve2

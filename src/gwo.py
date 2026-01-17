from __future__ import annotations

import math
import random
from typing import List, Tuple, Dict
from tqdm import tqdm

def _sigmoid(z: float) -> float:
    # numerically safer sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def grey_wolf_optimizer_binary_sharp(
    eval_fn,
    n_dim: int,
    pack_size: int = 18,
    iters: int = 25,
    seed: int = 0,
    w_clip: float = 6.0,          # wider than 4.0 but still stable
    beta_sigmoid: float = 1.0,    # base sharpness
    beta_sigmoid_end: float = 3.0,# increase sharpness over time
    accept_only_if_better: bool = True,
) -> Tuple[List[int], float, List[float]]:
    """
    Sharper Binary GWO:
      - Deterministic decoding of W -> X via sigmoid+threshold (no Bernoulli resampling)
      - Elitist replacement: only accept candidate if it improves (optional)
      - Internal caching of eval_fn calls
      - Annealed sigmoid sharpness beta(t): makes late-stage search more exploitative
    """

    rng = random.Random(seed)
    cache: Dict[Tuple[int, ...], float] = {}

    def eval_cached(x: List[int]) -> float:
        k = tuple(int(b) for b in x)
        if k in cache:
            return cache[k]
        v = float(eval_fn(x))
        cache[k] = v
        return v

    def decode(w: List[float], beta: float, tau: float = 0.5) -> List[int]:
        # deterministic mapping; "beta" sharpens sigmoid over time
        out = []
        for v in w:
            p = _sigmoid(beta * v)
            out.append(1 if p >= tau else 0)
        return out

    # init wolves in continuous space
    W = [[rng.uniform(-1.0, 1.0) for _ in range(n_dim)] for _ in range(pack_size)]

    # initial beta
    beta0 = beta_sigmoid
    beta1 = beta_sigmoid_end

    # deterministic initial X
    X = [decode(w, beta=beta0) for w in W]
    fits = [eval_cached(x) for x in X]

    curve: List[float] = []

    for t in tqdm(range(iters), desc="GWO+", leave=False):
        # leaders
        order = sorted(range(pack_size), key=lambda i: fits[i], reverse=True)
        alpha, beta_idx, delta = order[0], order[1], order[2]

        # a decreases 2 -> 0
        a = 2.0 - 2.0 * (t / max(1, iters - 1))

        # sharpen decoding over time (more exploit later)
        frac = t / max(1, iters - 1)
        beta_t = beta0 + (beta1 - beta0) * frac

        for i in range(pack_size):
            if i in (alpha, beta_idx, delta):
                continue

            new_w: List[float] = []
            for d in range(n_dim):
                # alpha
                r1, r2 = rng.random(), rng.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * W[alpha][d] - W[i][d])
                X1 = W[alpha][d] - A1 * D_alpha

                # beta
                r1, r2 = rng.random(), rng.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * W[beta_idx][d] - W[i][d])
                X2 = W[beta_idx][d] - A2 * D_beta

                # delta
                r1, r2 = rng.random(), rng.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * W[delta][d] - W[i][d])
                X3 = W[delta][d] - A3 * D_delta

                # average and clip
                x_d = (X1 + X2 + X3) / 3.0
                x_d = _clip(x_d, -w_clip, w_clip)
                new_w.append(x_d)

            cand_x = decode(new_w, beta=beta_t)
            cand_f = eval_cached(cand_x)

            if accept_only_if_better:
                # elitist replacement of each wolf
                if cand_f >= fits[i]:
                    W[i] = new_w
                    X[i] = cand_x
                    fits[i] = cand_f
            else:
                W[i] = new_w
                X[i] = cand_x
                fits[i] = cand_f

        curve.append(float(max(fits)))

    best_i = max(range(pack_size), key=lambda i: fits[i])
    return X[best_i][:], float(fits[best_i]), curve

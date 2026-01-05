from __future__ import annotations

import math
import random
from typing import List, Tuple


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def binary_pso(
    eval_fn,
    n_dim: int,
    swarm_size: int = 18,
    iters: int = 50,
    w: float = 0.7,
    c1: float = 1.4,
    c2: float = 1.4,
    seed: int = 0,
) -> Tuple[List[int], float, List[float]]:
    random.seed(seed)

    # positions (binary) and velocities (real)
    X = [[random.randint(0, 1) for _ in range(n_dim)] for _ in range(swarm_size)]
    V = [[random.uniform(-1, 1) for _ in range(n_dim)] for _ in range(swarm_size)]

    pbest = [x[:] for x in X]
    pbest_f = [eval_fn(x) for x in X]

    g_idx = max(range(swarm_size), key=lambda i: pbest_f[i])
    gbest = pbest[g_idx][:]
    gbest_f = pbest_f[g_idx]
    curve = [gbest_f]

    for _ in range(iters):
        for i in range(swarm_size):
            for d in range(n_dim):
                r1, r2 = random.random(), random.random()
                V[i][d] = (
                    w * V[i][d]
                    + c1 * r1 * (pbest[i][d] - X[i][d])
                    + c2 * r2 * (gbest[d] - X[i][d])
                )
                # update bit by sigmoid probability
                if random.random() < sigmoid(V[i][d]):
                    X[i][d] = 1
                else:
                    X[i][d] = 0

            f = eval_fn(X[i])
            if f > pbest_f[i]:
                pbest[i] = X[i][:]
                pbest_f[i] = f

        g_idx = max(range(swarm_size), key=lambda i: pbest_f[i])
        if pbest_f[g_idx] > gbest_f:
            gbest = pbest[g_idx][:]
            gbest_f = pbest_f[g_idx]

        curve.append(gbest_f)

    return gbest, gbest_f, curve

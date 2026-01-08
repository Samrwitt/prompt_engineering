from __future__ import annotations

import math
import random
from typing import List, Tuple
from tqdm import tqdm


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def binary_pso(
    eval_fn,
    n_dim: int,
    swarm_size: int = 20,
    iters: int = 50,
    w_start: float = 0.9,
    w_end: float = 0.4,
    c1: float = 1.4,
    c2: float = 1.4,
    seed: int = 0,
) -> Tuple[List[int], float, List[float]]:
    """
    Binary PSO with Inertia Weight Decay.
    
    Research Standards:
    - Linearly Decreasing Inertia Weight (LDIW) from w_start to w_end.
    - Standard sigmoid-based velocity clamp for binary space.
    """
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

    curve = [gbest_f]

    for t in tqdm(range(iters), desc="PSO Iterations", leave=False):
        # Linearly decay inertia w
        w = w_start - ((w_start - w_end) * t / iters)

        for i in range(swarm_size):
            for d in range(n_dim):
                r1, r2 = random.random(), random.random()
                
                # Velocity update
                V[i][d] = (
                    w * V[i][d]
                    + c1 * r1 * (pbest[i][d] - X[i][d])
                    + c2 * r2 * (gbest[d] - X[i][d])
                )
                
                # Position update (sigmoid binary)
                # s(v) represents probability of bit being 1
                if random.random() < sigmoid(V[i][d]):
                    X[i][d] = 1
                else:
                    X[i][d] = 0

            f = eval_fn(X[i])
            # Update personal best
            if f > pbest_f[i]:
                pbest[i] = X[i][:]
                pbest_f[i] = f

        # Update global best
        g_idx = max(range(swarm_size), key=lambda i: pbest_f[i])
        if pbest_f[g_idx] > gbest_f:
            gbest = pbest[g_idx][:]
            gbest_f = pbest_f[g_idx]

        curve.append(gbest_f)

    return gbest, gbest_f, curve

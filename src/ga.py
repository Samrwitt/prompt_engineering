from __future__ import annotations

import random
from typing import List, Tuple


def tournament_select(pop: List[List[int]], fits: List[float], k: int = 3) -> List[int]:
    idxs = random.sample(range(len(pop)), k=min(k, len(pop)))
    best = max(idxs, key=lambda i: fits[i])
    return pop[best][:]


def crossover(a: List[int], b: List[int]) -> List[int]:
    if len(a) <= 1:
        return a[:]
    p = random.randint(1, len(a) - 1)
    return a[:p] + b[p:]


def mutate(x: List[int], p: float = 0.05) -> List[int]:
    y = x[:]
    for i in range(len(y)):
        if random.random() < p:
            y[i] = 1 - y[i]
    return y


def genetic_algorithm(
    eval_fn,
    n_dim: int,
    pop_size: int = 16,
    generations: int = 40,
    mut_p: float = 0.05,
    seed: int = 0,
) -> Tuple[List[int], float, List[float]]:
    random.seed(seed)

    pop = [[random.randint(0, 1) for _ in range(n_dim)] for _ in range(pop_size)]
    fits = [eval_fn(ind) for ind in pop]

    best_idx = max(range(pop_size), key=lambda i: fits[i])
    best_x, best_f = pop[best_idx][:], fits[best_idx]
    curve = [best_f]

    for _ in range(generations):
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fits)
            p2 = tournament_select(pop, fits)
            child = crossover(p1, p2)
            child = mutate(child, p=mut_p)
            new_pop.append(child)

        pop = new_pop
        fits = [eval_fn(ind) for ind in pop]

        best_idx = max(range(pop_size), key=lambda i: fits[i])
        if fits[best_idx] > best_f:
            best_x, best_f = pop[best_idx][:], fits[best_idx]
        curve.append(best_f)

    return best_x, best_f, curve

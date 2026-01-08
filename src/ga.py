from __future__ import annotations

import random
from typing import List, Tuple
from tqdm import tqdm


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
    pop_size: int = 20,
    generations: int = 50,
    mut_p: float = 0.05,
    elitism: int = 2,  # Preserve top-2 best individuals
    seed: int = 0,
) -> Tuple[List[int], float, List[float]]:
    """
    Genetic Algorithm with Elitism.
    
    Research Standards:
    - Tournament selection
    - One-point crossover
    - Bit-flip mutation
    - ELITISM: Guarantee best solutions survive to next gen.
    """
    random.seed(seed)

    # Init population
    pop = [[random.randint(0, 1) for _ in range(n_dim)] for _ in range(pop_size)]
    fits = [eval_fn(ind) for ind in pop]

    # helper to sort pop by fitness
    # returns list of (fitness, individual_vector)
    def sort_pop(p, f):
        return sorted(zip(f, p), key=lambda pair: pair[0], reverse=True)

    # Initial best
    sorted_pop = sort_pop(pop, fits)
    best_f, best_x = sorted_pop[0]
    curve = [best_f]

    curve = [best_f]

    for _ in tqdm(range(generations), desc="GA Generations", leave=False):
        new_pop = []
        
        # 1. Elitism: Copy best k directly
        # Re-sort current pop just to be safe/explicit
        sorted_pop = sort_pop(pop, fits)
        
        # Update global best if improved
        if sorted_pop[0][0] > best_f:
            best_f, best_x = sorted_pop[0]

        # Carry over elites
        for i in range(min(elitism, pop_size)):
            new_pop.append(sorted_pop[i][1][:])
            
        # 2. Breed the rest
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fits)
            p2 = tournament_select(pop, fits)
            child = crossover(p1, p2)
            child = mutate(child, p=mut_p)
            new_pop.append(child)

        pop = new_pop
        fits = [eval_fn(ind) for ind in pop]

        # Record generation best (checking if we found a new global best implicitly)
        curr_best_f = max(fits)
        if curr_best_f > best_f:
             # Just in case we found it in normal offspring
             best_idx = fits.index(curr_best_f)
             best_x = pop[best_idx][:]
             best_f = curr_best_f
             
        curve.append(best_f)

    return best_x, best_f, curve

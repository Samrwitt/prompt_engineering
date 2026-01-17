import random
from typing import Callable, List, Tuple
from tqdm import tqdm

def random_search(
    eval_fn: Callable[[List[int]], float],
    n_dim: int,
    seed: int = 0,
    iters: int = 100,
) -> Tuple[List[int], float, List[Tuple[int, float]]]:
    """
    Randomly samples x in {0,1}^n_dim.
    """
    rng = random.Random(seed)
    best_val = -1.0
    best_x = [0] * n_dim
    curve = []
    
    for i in tqdm(range(iters), desc="Random Search", leave=False):
        x = [rng.randint(0, 1) for _ in range(n_dim)]
        val = eval_fn(x)
        if val > best_val:
            best_val = val
            best_x = x[:]
        curve.append((i, best_val))
        
    return best_x, best_val, curve


def greedy_add_one(
    eval_fn: Callable[[List[int]], float],
    n_dim: int,
    seed: int = 0,
    restarts: int = 1,
) -> Tuple[List[int], float, List[Tuple[int, float]]]:
    """
    Greedy add-one-best (Next Ascent style for efficiency).
    """
    rng = random.Random(seed)
    best_global_val = -1.0
    best_global_x = [0] * n_dim
    curve = []

    for r in range(restarts):
        if r == 0:
            current_x = [0] * n_dim
        else:
            current_x = [rng.randint(0, 1) for _ in range(n_dim)]
            
        current_val = eval_fn(current_x)
        if current_val > best_global_val:
            best_global_val = current_val
            best_global_x = current_x[:]
            
        improved = True
        with tqdm(total=n_dim, desc=f"Greedy R{r}", leave=False) as pbar:
            while improved:
                improved = False
                candidates = [i for i, bit in enumerate(current_x) if bit == 0]
                rng.shuffle(candidates)
                
                for i in candidates:
                    neighbor = current_x[:]
                    neighbor[i] = 1
                    val = eval_fn(neighbor)
                    pbar.update(1)
                    
                    if val > current_val:
                        current_val = val
                        current_x = neighbor[:]
                        improved = True
                        if val > best_global_val:
                            best_global_val = val
                            best_global_x = current_x[:]
                        break 
                if not improved:
                    break

    return best_global_x, best_global_val, curve


def hill_climb_bit_flip(
    eval_fn: Callable[[List[int]], float],
    n_dim: int,
    seed: int = 0,
    restarts: int = 2,
    max_steps: int = 50,
) -> Tuple[List[int], float, List[Tuple[int, float]]]:
    """
    Standard Hill Climbing with restarts.
    """
    rng = random.Random(seed)
    best_global_val = -1.0
    best_global_x = [0] * n_dim
    curve = []

    for r in range(restarts):
        current_x = [rng.randint(0, 1) for _ in range(n_dim)]
        current_val = eval_fn(current_x)
        
        if current_val > best_global_val:
            best_global_val = current_val
            best_global_x = current_x[:]
            
        for _ in tqdm(range(max_steps), desc=f"HC R{r}", leave=False):
            idx = rng.randint(0, n_dim - 1)
            neighbor = current_x[:]
            neighbor[idx] = 1 - neighbor[idx]
            
            val = eval_fn(neighbor)
            if val >= current_val:
                current_val = val
                current_x = neighbor[:]
                if val > best_global_val:
                    best_global_val = val
                    best_global_x = current_x[:]
                
    return best_global_x, best_global_val, curve

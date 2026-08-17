"""Optimizer registry used by the experiment runner, CLI, and dashboard."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from src.de import differential_evolution_binary_sharp
from src.de_sa import hybrid_de_sa
from src.gwo import grey_wolf_optimizer_binary_sharp
from src.sa import simulated_annealing_sharp
from src.simple_opt import greedy_add_one, hill_climb_bit_flip, random_search

AlgoFn = Callable[..., Tuple[List[int], float, Any]]


def fixed_baseline(eval_fn, n_dim, seed, x_fixed, **kwargs):
    full_x = list(x_fixed)
    if len(full_x) < n_dim:
        full_x += [0] * (n_dim - len(full_x))
    val = eval_fn(full_x)
    return full_x, val, [val]


def build_methods(n_blocks: int, cfg) -> List[Tuple[str, AlgoFn, Dict[str, Any]]]:
    ones = [1] * n_blocks
    zeros = [0] * n_blocks
    return [
        ("BASELINE_ALL", fixed_baseline, dict(x_fixed=ones)),
        ("BASELINE_NONE", fixed_baseline, dict(x_fixed=zeros)),
        ("RANDOM", random_search, dict(iters=min(50, cfg.max_llm_calls))),
        ("GREEDY", greedy_add_one, dict(restarts=1)),
        ("SA+", simulated_annealing_sharp, dict(
            iters=cfg.sa_iters, t0=1.0, cooling=0.97, stagnation_reheat=12
        )),
        ("DE", differential_evolution_binary_sharp, dict(
            pop_size=cfg.pop_size, iters=cfg.de_iters, F=0.7, CR=0.8
        )),
        ("GWO", grey_wolf_optimizer_binary_sharp, dict(
            pack_size=cfg.pop_size, iters=cfg.gwo_iters
        )),
        ("HYBRID_DE_SA", hybrid_de_sa, dict(
            pop_size=cfg.pop_size,
            de_iters=cfg.hybrid_de,
            sa_iters=cfg.hybrid_sa,
        )),
        ("HILL_CLIMB", hill_climb_bit_flip, dict(restarts=1, max_steps=min(30, cfg.max_llm_calls))),
    ]


METHOD_HELP = {
    "BASELINE_ALL": "Include every instruction block, no few-shot demos.",
    "BASELINE_NONE": "Empty prompt — model sees only the question.",
    "RANDOM": "Uniform samples from the binary genome.",
    "GREEDY": "Forward selection: add the next bit that improves fitness.",
    "SA+": "Simulated Annealing with adaptive neighborhood, tabu cache, and reheating.",
    "DE": "Binary Differential Evolution with annealed sigmoid decoding.",
    "GWO": "Binary Grey Wolf Optimizer with elitist replacement.",
    "HYBRID_DE_SA": "Population DE search, then SA local refinement.",
    "HILL_CLIMB": "Random-restart bit-flip hill climbing.",
}

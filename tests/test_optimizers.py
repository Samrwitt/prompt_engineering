from src.de import differential_evolution_binary_sharp
from src.gwo import grey_wolf_optimizer_binary_sharp
from src.sa import simulated_annealing_sharp
from src.simple_opt import random_search


def _onemax(x):
    return sum(int(b) for b in x) / len(x)


def test_sa_improves_onemax():
    best_x, best_f, curve = simulated_annealing_sharp(_onemax, n_dim=12, iters=40, seed=0)
    assert len(best_x) == 12
    assert best_f >= 0.5
    assert curve[-1] >= curve[0]


def test_de_returns_binary_vector():
    best_x, best_f, _ = differential_evolution_binary_sharp(_onemax, n_dim=8, pop_size=8, iters=6, seed=1)
    assert all(b in (0, 1) for b in best_x)
    assert 0.0 <= best_f <= 1.0


def test_gwo_returns_binary_vector():
    best_x, best_f, _ = grey_wolf_optimizer_binary_sharp(_onemax, n_dim=8, pack_size=8, iters=6, seed=2)
    assert all(b in (0, 1) for b in best_x)
    assert best_f >= 0.0


def test_random_search_tracks_best():
    best_x, best_f, curve = random_search(_onemax, n_dim=10, seed=3, iters=20)
    assert best_f == max(p[1] for p in curve)
    assert len(best_x) == 10

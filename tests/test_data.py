from src.generate_data import generate_arithmetic, generate_boolean
from src.tasks import solve_arithmetic, solve_boolean


def test_boolean_solver_matches_python_precedence():
    assert solve_boolean("Evaluate: True AND False OR True.") == "yes"
    assert solve_boolean("Evaluate: NOT (True OR False) AND True.") == "no"


def test_generated_boolean_labels_are_solvable():
    rows = generate_boolean(40, seed=0)
    assert len(rows) == 40
    assert {r["a"] for r in rows} <= {"yes", "no"}
    for row in rows:
        assert solve_boolean(row["q"]) == row["a"]


def test_generated_arithmetic_labels_match_ops():
    rows = generate_arithmetic(40, seed=0)
    assert len(rows) == 40
    for row in rows:
        parsed = solve_arithmetic(row["q"])
        if parsed is not None:
            assert parsed == row["a"]
        else:
            assert row["a"].lstrip("-").isdigit()

"""Generate verified logic and arithmetic datasets."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from src.tasks import solve_arithmetic, solve_boolean

ATOMS = ["True", "False"]


def save_jsonl(path: str, data: List[Dict[str, str]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def _atom(rng: random.Random) -> str:
    token = rng.choice(ATOMS)
    if rng.random() < 0.4:
        return f"NOT {token}"
    return token


def _expr(rng: random.Random, depth: int) -> str:
    if depth <= 0:
        return _atom(rng)
    left = _expr(rng, depth - 1)
    right = _expr(rng, depth - 1)
    op = rng.choice(["AND", "OR"])
    body = f"({left}) {op} ({right})"
    if rng.random() < 0.3:
        return f"NOT {body}"
    return body


def generate_boolean(n: int = 100, seed: int = 42) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    seen = set()
    rows: List[Dict[str, str]] = []
    # Keep a few hand-written nested cases first.
    seeds = [
        "True AND False OR True",
        "NOT (True OR False) AND True",
        "NOT False AND (False OR True) AND True",
        "True OR NOT False AND False OR NOT True",
        "(True AND False) OR (False AND True) OR True",
        "NOT (True AND True) AND (False OR False)",
        "True AND (True OR NOT True) AND False",
        "NOT True OR NOT False AND True OR False",
        "((True OR False) AND NOT False) OR False",
        "NOT (False AND NOT True OR True) AND True",
    ]
    for expr in seeds:
        ans = solve_boolean(expr)
        if ans:
            q = f"Evaluate: {expr}."
            rows.append({"q": q, "a": ans})
            seen.add(q)

    guard = 0
    while len(rows) < n and guard < n * 40:
        guard += 1
        depth = rng.choice([1, 1, 2, 2, 3])
        expr = _expr(rng, depth)
        ans = solve_boolean(expr)
        if not ans:
            continue
        q = f"Evaluate: {expr}."
        if q in seen:
            continue
        seen.add(q)
        rows.append({"q": q, "a": ans})
    if len(rows) < n:
        raise RuntimeError(f"Only generated {len(rows)} unique boolean items")
    return rows[:n]


def generate_arithmetic(n: int = 100, seed: int = 42) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    seen = set()
    while len(rows) < n:
        kind = rng.choice(["plain", "plain", "plain", "word"])
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        op = rng.choice(["+", "-", "*"])
        if kind == "plain":
            q = f"What is {a} {op} {b}?"
        elif op == "+":
            q = f"A box had {a} items. After adding {b} more, how many are there now?"
        elif op == "-":
            q = f"What is the result of subtracting {b} from {a}?"
        else:
            q = f"What is {a} times {b}?"
        if q in seen:
            continue
        if op == "+":
            ans = str(a + b)
        elif op == "-":
            ans = str(a - b)
        else:
            ans = str(a * b)
        # Cross-check the parseable subset.
        parsed = solve_arithmetic(q)
        if parsed is not None and parsed != ans:
            raise AssertionError(f"solver mismatch for {q}: {parsed} vs {ans}")
        seen.add(q)
        rows.append({"q": q, "a": ans})
    return rows


def main() -> None:
    logic = generate_boolean(100)
    save_jsonl("data/bbh_boolean_expressions.jsonl", logic)
    print(f"Wrote {len(logic)} boolean items")

    arith = generate_arithmetic(100)
    save_jsonl("data/arithmetic.jsonl", arith)
    print(f"Wrote {len(arith)} arithmetic items")

    yes = sum(1 for r in logic if r["a"] == "yes")
    print(f"Boolean class balance: yes={yes} no={len(logic) - yes}")


if __name__ == "__main__":
    main()

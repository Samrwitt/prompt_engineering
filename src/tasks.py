"""Deterministic solvers used by data generation and the mock LLM."""
from __future__ import annotations

import re
from typing import Optional

_ARITH_RE = re.compile(r"what is\s+(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*\??", re.I)
_SAFE_BOOL = re.compile(r"^[TrueFalsandornot() ]+$")


def solve_boolean(question: str) -> Optional[str]:
    q = question.strip()
    q = re.sub(r"^evaluate:\s*", "", q, flags=re.I)
    q = re.sub(r"\.\s*$", "", q)
    q = re.sub(r"answer yes or no only\.?$", "", q, flags=re.I)
    q = re.sub(r"use precedence.*$", "", q, flags=re.I)
    expr = q.strip()
    py = (
        expr.replace("True", " True ")
        .replace("False", " False ")
        .replace("AND", " and ")
        .replace("OR", " or ")
        .replace("NOT", " not ")
    )
    py = re.sub(r"\s+", " ", py).strip()
    if not _SAFE_BOOL.fullmatch(py):
        return None
    try:
        result = eval(py, {"__builtins__": {}}, {})  # noqa: S307 — literal True/False/and/or/not only
    except Exception:
        return None
    if not isinstance(result, bool):
        return None
    return "yes" if result else "no"


def solve_arithmetic(question: str) -> Optional[str]:
    m = _ARITH_RE.search(question)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op == "*":
        return str(a * b)
    if op == "/" and b != 0:
        return str(a // b)
    return None

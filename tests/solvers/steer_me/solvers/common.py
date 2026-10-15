"""Shared solver framework.

A solver re-derives the answer of one question from its *text only* (never from filler code) and reports,
for every option, whether that option equals the independently computed answer.

    @register("element_name")
    def solve(q: str, options: list[str], qtype: str) -> Result: ...

Return Result(status="unparsed") when the text does not fit the solver (that is not evidence of a bug),
Result(status="undefined", note=...) when the question has no well-defined answer (e.g. no finite optimum),
and otherwise Result(status="ok", expected=<value shown in reports>, correct=[bool per option]).

Tolerances: numbers shown in questions are rounded to 2 dp while fillers compute options from the unrounded
values, so numeric comparisons use |a-b| <= max(ABS_TOL, REL_TOL*|b|).  `closest` records which option is
nearest to the expected value, so "marked == closest" separates rounding noise from wrong answers.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

REL_TOL = 0.02
ABS_TOL = 0.011

NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
NUM_RE = re.compile(NUM)


@dataclass
class Result:
    status: str = "unparsed"              # ok | unparsed | undefined
    expected: object = None               # the independently computed answer (for reports)
    correct: list[bool] | None = None     # per option: equals the expected answer?
    closest: int | None = None            # index of the option nearest to expected (numeric solvers)
    note: str = ""


SOLVERS: dict[str, Callable[[str, list[str], str], Result]] = {}


def register(*elements: str):
    def deco(fn):
        for e in elements:
            if e in SOLVERS:
                raise ValueError(f"duplicate solver for {e}")
            SOLVERS[e] = fn
        return fn
    return deco


# ---------------------------------------------------------------- numbers
def nums(s: str) -> list[float]:
    return [float(x) for x in NUM_RE.findall(s)]


def first_num(pattern: str, q: str, flags=re.IGNORECASE | re.DOTALL) -> float | None:
    """First capture group of `pattern` (use {N} for a number group) as float."""
    m = re.search(pattern.replace("{N}", f"({NUM})"), q, flags)
    return float(m.group(1)) if m else None


def close(a: float, b: float, rel: float = REL_TOL, abs_: float = ABS_TOL) -> bool:
    if a is None or b is None or not all(map(math.isfinite, (a, b))):
        return False
    return abs(a - b) <= max(abs_, rel * abs(b))


def to_float(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def numeric_result(expected: float, options: Sequence[str], rel=REL_TOL, abs_=ABS_TOL, note="") -> Result:
    """Options are plain numbers; mark those within tolerance of expected."""
    vals = [to_float(o) for o in options]
    if any(v is None for v in vals) or expected is None or not math.isfinite(expected):
        return Result(status="unparsed", note=note or "non-numeric option or expected")
    corr = [close(v, expected, rel, abs_) for v in vals]
    closest = min(range(len(vals)), key=lambda i: abs(vals[i] - expected))
    return Result("ok", round(expected, 6), corr, closest, note)


def vector_result(expected: Sequence[float], options: Sequence[str], parse=None, rel=REL_TOL, abs_=ABS_TOL,
                  note="") -> Result:
    """Options are tuples of numbers ('(1.2, 3.4)', '[1, 2]', 'C1 = 1, C2 = 2'); all components must match."""
    parse = parse or nums
    vecs = [parse(o) for o in options]
    if any(len(v) != len(expected) for v in vecs):
        return Result(status="unparsed", note=note or "option arity mismatch")
    corr = [all(close(a, b, rel, abs_) for a, b in zip(v, expected)) for v in vecs]
    dist = [sum(abs(a - b) / max(1.0, abs(b)) for a, b in zip(v, expected)) for v in vecs]
    closest = min(range(len(vecs)), key=dist.__getitem__)
    return Result("ok", tuple(round(x, 6) for x in expected), corr, closest, note)


def text_result(expected: str, options: Sequence[str], norm=lambda s: s.strip().lower(), note="") -> Result:
    corr = [norm(o) == norm(expected) for o in options]
    return Result("ok", expected, corr, corr.index(True) if any(corr) else None, note)


def match_result(expected, options: Sequence[str], same: Callable[[str], bool | None], note="") -> Result:
    """Generic: `same(option)` returns True/False, or None if that option could not be parsed."""
    corr = [same(o) for o in options]
    if any(c is None for c in corr):
        return Result(status="unparsed", note=note or "option not parseable")
    return Result("ok", expected, [bool(c) for c in corr], corr.index(True) if any(corr) else None, note)


# ---------------------------------------------------------------- functional forms
def cobb_douglas_exponents(q: str, var1: str = "L", var2: str = "K") -> tuple[float, float, float] | None:
    """Parse 'A*L^a K^b' / 'L^aK^b' / 'x^a y^b' -> (A, a, b)."""
    m = re.search(rf"({NUM})?\s*\*?\s*{var1}\^\(?({NUM})\)?\s*\*?\s*{var2}\^\(?({NUM})\)?", q)
    if not m:
        return None
    A = float(m.group(1)) if m.group(1) else 1.0
    return A, float(m.group(2)), float(m.group(3))

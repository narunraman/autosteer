"""
Independent re-implementations of the economics behind each element.

These deliberately do not import any filler code: they parse the function
strings that are shown in the question and recompute the answer from the
displayed (rounded) parameters.
"""
from __future__ import annotations

import math
import re

NUMBER = r'\d+(?:\.\d+)?(?:e[-+]?\d+)?'


# ----------------------------------------------------------------------
# Polynomials / monomials in named variables, e.g. "5.49L + 7.15K - 0.61L^2 + 4.24LK"
# or "L^0.5K^0.3" or "3.1x^2 + 2x + 1".
# ----------------------------------------------------------------------

def parse_terms(expr: str, variables: str) -> list[tuple[float, dict[str, float]]]:
    """Parse a sum of monomials into [(coef, {var: power})]."""
    expr = expr.replace(' ', '').replace('*', '')
    # split keeping signs
    pieces = re.findall(r'[+-]?[^+-]+(?:e[-+]\d+)?', expr)
    terms = []
    var_pat = re.compile(rf'([{variables}])(?:\^({NUMBER}))?')
    for piece in pieces:
        sign = -1.0 if piece.startswith('-') else 1.0
        piece = piece.lstrip('+-')
        m = re.match(NUMBER, piece)
        coef = float(m.group(0)) if m else 1.0
        rest = piece[m.end():] if m else piece
        powers: dict[str, float] = {}
        pos = 0
        while pos < len(rest):
            vm = var_pat.match(rest, pos)
            if not vm:
                raise ValueError(f'cannot parse term {piece!r} in {expr!r}')
            powers[vm.group(1)] = powers.get(vm.group(1), 0.0) + (float(vm.group(2)) if vm.group(2) else 1.0)
            pos = vm.end()
        terms.append((sign * coef, powers))
    return terms


def eval_terms(terms, values: dict[str, float]) -> float:
    total = 0.0
    for coef, powers in terms:
        v = coef
        for var, p in powers.items():
            v *= values[var] ** p
        total += v
    return total


def diff_terms(terms, var: str):
    out = []
    for coef, powers in terms:
        p = powers.get(var, 0.0)
        if p == 0:
            continue
        new = dict(powers)
        new[var] = p - 1
        if new[var] == 0:
            del new[var]
        out.append((coef * p, new))
    return out


def poly_x(expr: str) -> dict[int, float]:
    """Parse a polynomial in x into {power: coef}."""
    out: dict[int, float] = {}
    for coef, powers in parse_terms(expr, 'x'):
        k = int(round(powers.get('x', 0)))
        out[k] = out.get(k, 0.0) + coef
    return out


# ----------------------------------------------------------------------
# Economics
# ----------------------------------------------------------------------

def cobb_douglas_marshallian(alpha: float, beta: float, income: float, good: int) -> float:
    """Coefficient c in x_i = c / p_i for U = x^alpha y^beta."""
    share = (alpha if good == 1 else beta) / (alpha + beta)
    return share * income


def elasticity_arc_simple(p0, p1, q0, q1) -> float:
    """(%dQ)/(%dP) with the starting point as base."""
    return ((q1 - q0) / q0) / ((p1 - p0) / p0)


def unit_cost_price_change_productivity(pct: float) -> float:
    """% change in P = c(w, r)/A when A changes by pct %."""
    return (1 / (1 + pct / 100) - 1) * 100


def unit_cost_price_change_wage(pct: float, alpha: float) -> float:
    """% change in P = A^-1 (w/alpha)^alpha (r/(1-alpha))^(1-alpha) when w changes by pct %."""
    return ((1 + pct / 100) ** alpha - 1) * 100


def cd_short_run_profit_max(price, a, b, capital, wage):
    """max_L p L^a K^b - w L for 0 < a < 1. Returns (L*, profit)."""
    labor = (a * price * capital ** b / wage) ** (1 / (1 - a))
    profit = price * labor ** a * capital ** b - wage * labor
    return labor, profit


def leontief_short_run_profit_max(price, cl, ck, v, capital, wage):
    """max_L p min(L/cl, K/ck)^v - w L with K fixed, 0 < v < 1. Returns (L*, profit)."""
    cap_labor = cl * capital / ck  # labor beyond this adds no output
    # interior optimum of p (L/cl)^v - w L
    interior = (v * price / (wage * cl ** v)) ** (1 / (1 - v))
    labor = min(interior, cap_labor)
    profit = price * (labor / cl) ** v - wage * labor
    return labor, profit


def quadratic_roots(a, b, c):
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    r = math.sqrt(disc)
    if a == 0:
        return [-c / b]
    return [(-b + r) / (2 * a), (-b - r) / (2 * a)]

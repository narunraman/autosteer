"""Rounding ties (review, October 2026).

A key printed at 2 decimals is ambiguous when its exact value lies on a rounding tie: 1.31 x 24.5 = 32.095 exactly,
binary floating point printed 32.09 and half-up rounding gives 32.10. The fillers redraw such values instead of
choosing a rounding rule, so every reasonable rounding of the exact value gives the printed key.
"""
from __future__ import annotations


def near_tie(x: float, places: int = 2, tol: float = 1e-6) -> bool:
    """True if x is within tol (in units of the last place) of a rounding tie at `places` decimals."""
    frac = (abs(float(x)) * 10 ** places) % 1.0
    return abs(frac - 0.5) < tol

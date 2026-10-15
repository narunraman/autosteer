"""Production / cost elements: comp_stat_leonteif, comp_stat_cobb, average_cost, marginal_cost, returns_to_scale.

All answers except returns_to_scale are power functions of one variable, shown as options like
'-22.53C^-3', '0.26Q^0.82+32.1/Q' or '6.25Q^0.12'.  Each option is parsed into (coefficient, exponent, F)
where F is the optional '+F/Q' fixed-cost term, and compared with the answer derived from the question text.

Tolerance: the parameters shown in the question are the exact values (the fillers round them to 2 dp *before*
using them), so the expected answer is computed from the displayed numbers without perturbation.  An option
number matches when it equals the expected value up to its own 2-dp display rounding (+-0.005).

If the correct coefficient displays as 0.0 (|coef| < 0.005) and that does not single out one option (several
~0 options, or duplicates), the question is returned as "undefined" (counted separately by the runner).
Options in scientific notation ('-3.52e-05C^-3') are compared relative to their mantissa.
"""
from __future__ import annotations

import math
import re

from .common import NUM, Result, register, text_result


# ------------------------------------------------------------------------------------------ helpers
def _half_ulp(s: str) -> float:
    """Half a unit in the last displayed digit, never finer than 2 dp ('1.6' was 1.60 -> 0.005).

    Scientific notation ('-4.04e+66') is taken relative to its mantissa: 0.005e66.
    """
    s = s.strip().lstrip("+-").lower()
    mant, _, ex = s.partition("e")
    dec = len(mant.split(".")[1]) if "." in mant else 0
    return 0.5 * 10 ** (-max(2, dec)) * 10 ** (int(ex) if ex else 0)


# option: [coef] VAR [^exp] [+ F/VAR]
_OPT = re.compile(rf"^({NUM})?\*?([A-Za-z])(?:\^\(?({NUM})\)?)?(?:\+({NUM})/\2)?$")


def _parse_opt(o: str):
    """'0.26Q^0.82+32.1/Q' -> dict(coef='0.26', exp='0.82', F='32.1', var='Q'); None if not a power term."""
    m = _OPT.match(re.sub(r"\s+", "", o))
    if not m:
        return None
    return {"coef": m.group(1) or "1", "var": m.group(2), "exp": m.group(3) or "1", "F": m.group(4)}


def _match(shown: str, true: float) -> bool:
    """Does the displayed option number equal `true` up to its own display rounding (+ float noise)?

    A displayed zero keeps its sign ('-0.0' vs '0.0'), so only the correctly signed zero can match.
    """
    x = float(shown)
    if x == 0 and true != 0 and math.copysign(1, x) != math.copysign(1, true):
        return False
    return abs(x - true) <= _half_ulp(shown) * (1 + 1e-6) + 1e-9 * abs(true)


def _power_result(coef: float, exp: float, options: list[str], F: str | None = None, note=""):
    """Score options against the expected answer coef * X^exp [+ F/X]."""
    if not (math.isfinite(coef) and math.isfinite(exp)):
        return Result("undefined", note="expected answer not finite")
    expected = f"{coef:.6g}X^{exp:.6g}" + (f"+{F}/X" if F else "")
    parsed = [_parse_opt(o) for o in options]
    if any(p is None for p in parsed):
        return Result("unparsed", note="option not a power term")
    corr, dist = [], []
    for p in parsed:
        f_ok = (p["F"] is None) == (F is None) and (F is None or math.isclose(float(p["F"]), float(F), abs_tol=1e-6))
        corr.append(f_ok and _match(p["coef"], coef) and _match(p["exp"], exp))
        dist.append((0 if f_ok else 10) + abs(float(p["exp"]) - exp) + abs(float(p["coef"]) - coef) / max(1.0, abs(coef)))
    closest = min(range(len(options)), key=dist.__getitem__)
    if abs(coef) < 0.005 and sum(corr) != 1:
        # the answer displays as (-)0.0 and cannot be told apart from other ~0 options
        return Result("undefined", expected=expected, note="correct coefficient rounds to 0.0 at 2 dp; ambiguous")
    return Result("ok", expected, corr, closest, note)


_NUMTOK = re.compile(rf"(?<![\w.^]){NUM}(?![\w^])")


def _numstrs(s: str) -> list[str]:
    """Stand-alone numbers (not part of tokens like '3D'), trailing sentence dots stripped."""
    return [x.rstrip(".") for x in _NUMTOK.findall(s)]


def _remove(s: str, span: tuple[int, int]) -> str:
    return s[:span[0]] + " " + s[span[1]:]


_CD = re.compile(rf"(?<![\w.])({NUM})?\s*\*?\s*L\s*\^\s*\(?({NUM})\)?\s*\*?\s*K\s*\^\s*\(?({NUM})\)?")


# ------------------------------------------------------------------------------------------ comp_stat_leonteif
_LEON = re.compile(rf"min\s*\(\s*L\s*/\s*({NUM})\s*,\s*K\s*/\s*({NUM})\s*\)\s*(?:\^\s*\(?\s*({NUM})\s*\)?)?", re.I)


@register("comp_stat_leonteif")
def comp_stat_leonteif(q, options, qtype):
    """Price p, output f = min(L/a, K/b)^r, labor cost w*L, capital cost fixed (sunk).

    Assumption: capital is not a binding constraint (its level is never given), so the firm chooses L on the
    labor branch: max p (L/a)^r - wL  ->  L*(w) = a (p r / (a w))^(1/(1-r)).
    Hence dL*/dw = -(1/(1-r)) a (p r / a)^(1/(1-r)) w^(-1/(1-r) - 1); for r = 0.5 this is -p^2/(2a) w^-3.
    The options write the wage as C (the text calls it w / m / v / ...).
    The price is the only stand-alone number in the text outside the production function.
    """
    m = _LEON.search(q)
    if not m:
        return Result("unparsed", note="no min(L/a,K/b)^r")
    a, r = float(m.group(1)), float(m.group(3) or 1)
    rest = _numstrs(_remove(q, m.span()))
    if len(rest) != 1:
        return Result("unparsed", note=f"expected exactly one price, got {rest}")
    p = float(rest[0])
    if r >= 1:
        return Result("undefined", note=f"r={r} >= 1: no interior profit maximum")
    e = 1 / (1 - r)
    return _power_result(-e * a * (p * r / a) ** e, -e - 1, options)


# ------------------------------------------------------------------------------------------ comp_stat_cobb
_KFIX = [rf"\bK\s*(?:=|is|being|of|equals|equal to)\s*({NUM})",
         rf"(?:fixed|constant|set|held)\s+(?:at|to)\s+(?:a\s+(?:level|value)\s+of\s+)?({NUM})"]


_PRICE_WORDS = re.compile(r"pric|sold|sell|charg|valued|market|revenue|fee|per unit|each unit", re.I)


def _cobb_k_p(text: str):
    """(k, p) as strings from the text with the production function removed, or None.

    First look for an explicit 'K = k' / 'fixed at k'; otherwise, with exactly two numbers left, the one in a
    sentence with price words is p and the other k.
    """
    for pat in _KFIX:
        m = re.search(pat, text, re.I)
        if m:
            rest = _numstrs(_remove(text, m.span(1)))
            return (m.group(1), rest[0]) if len(rest) == 1 else None
    found = [(x, bool(_PRICE_WORDS.search(sent)))
             for sent in re.split(r"(?<=[.!?])\s+", text) for x in _numstrs(sent)]
    if len(found) == 2 and found[0][1] != found[1][1]:
        (x, xp), (y, _) = found
        return (y, x) if xp else (x, y)
    return None


@register("comp_stat_cobb")
def comp_stat_cobb(q, options, qtype):
    """Type marginal_cost_labor.  Price p, output f = L^a K^b with K fixed at k, labor cost C*L.

    FOC p a L^(a-1) k^b = C  ->  L*(C) = (p a k^b / C)^(1/(1-a))   (a < 1 needed for an interior maximum),
    dL*/dC = -(1/(1-a)) (p a k^b)^(1/(1-a)) C^(-1/(1-a) - 1).
    k is the number after 'K =' / 'fixed at' (else the number not in a price sentence); p is the other.
    """
    m = _CD.search(q)
    if not m:
        return Result("unparsed", note="no L^aK^b")
    A, a, b = float(m.group(1) or 1), float(m.group(2)), float(m.group(3))
    kp = _cobb_k_p(_remove(q, m.span()))
    if kp is None:
        return Result("unparsed", note="capital level / price not identified")
    k, p = map(float, kp)
    if a >= 1 or a <= 0:
        return Result("undefined", note=f"a={a}: no interior profit maximum in L")
    e = 1 / (1 - a)
    return _power_result(-e * (p * A * a * k ** b) ** e, -e - 1, options)


# ------------------------------------------------------------------------------------------ cost curves
_UNIT = re.compile(rf"(?<![\w.^])({NUM})\s*\*?\s*L\b(?!\s*\^)")


def _cost_params(q: str):
    """(F, A, a, b, k, c) from an average/marginal-cost question, or a note string if it does not fit.

    Every template states the fixed cost F before the capital level k (verified on 1,500 fresh draws), so after
    removing the production function and the labor cost 'cL' the two remaining numbers are F, k in order.
    """
    m = _CD.search(q)
    if not m:
        return "no L^aK^b"
    A, a, b = float(m.group(1) or 1), float(m.group(2)), float(m.group(3))
    # Review (October 2026): the questions end with a sentence that states the model ("... K as fixed at k;
    # the variable cost is $c per unit of L (cL in total) and the fixed cost is F."); read the numbers from it.
    conv = re.search(r"K as fixed at (\d+(?:\.\d+)?); the variable cost is \$(\d+(?:\.\d+)?) per unit of L "
                     r"\(\S+ in total\) and the fixed cost is (\d+(?:\.\d+)?)\.", q)
    if conv:
        return float(conv.group(3)), A, a, b, float(conv.group(1)), float(conv.group(2))
    t = _remove(q, m.span())
    mc = list(_UNIT.finditer(t))
    if len(mc) != 1:
        return f"labor unit cost 'cL' found {len(mc)} times"
    c = float(mc[0].group(1))
    rest = _numstrs(_remove(t, mc[0].span()))
    if len(rest) != 2:
        return f"expected fixed cost and capital, got {rest}"
    return rest[0], A, a, b, float(rest[1]), c


def _var_cost_coef(A, a, b, k, c):
    """Variable cost c L(Q) with L(Q) = (Q / (A k^b))^(1/a):  c (A k^b)^(-1/a) Q^(1/a)."""
    return c * (A * k ** b) ** (-1 / a)


@register("average_cost")
def average_cost(q, options, qtype):
    """Fixed cost F, output Q = L^a K^b with K fixed at k, labor cost c*L.

    Inverting the production function, L(Q) = (Q / k^b)^(1/a), so C(Q) = F + c k^(-b/a) Q^(1/a) and
    AC(Q) = C(Q)/Q = c k^(-b/a) Q^(1/a - 1) + F/Q.
    """
    r = _cost_params(q)
    if isinstance(r, str):
        return Result("unparsed", note=r)
    F, A, a, b, k, c = r
    if a == 0:
        return Result("undefined", note="a = 0: output does not depend on labor")
    return _power_result(_var_cost_coef(A, a, b, k, c), 1 / a - 1, options, F=F)


@register("marginal_cost")
def marginal_cost(q, options, qtype):
    """Same setting as average_cost:  MC(Q) = C'(Q) = (c/a) k^(-b/a) Q^(1/a - 1)  (the fixed cost drops out)."""
    r = _cost_params(q)
    if isinstance(r, str):
        return Result("unparsed", note=r)
    _F, A, a, b, k, c = r
    if a == 0:
        return Result("undefined", note="a = 0: output does not depend on labor")
    return _power_result(_var_cost_coef(A, a, b, k, c) / a, 1 / a - 1, options)


# ------------------------------------------------------------------------------------------ returns_to_scale
_LIN = re.compile(rf"(?<![\w.^])({NUM})?\s*\*?\s*L\s*\+\s*({NUM})?\s*\*?\s*K\b(?!\s*\^)")


@register("returns_to_scale")
def returns_to_scale(q, options, qtype):
    """Cobb-Douglas A L^a K^b is homogeneous of degree a+b: increasing if a+b > 1, constant if = 1,
    decreasing if < 1.  A linear function aL + bK is homogeneous of degree 1: constant returns."""
    m = _CD.search(q)
    if m:
        s = float(m.group(2)) + float(m.group(3))  # exponents are shown exactly as used
        rts = "Increasing" if s > 1 + 1e-9 else "Decreasing" if s < 1 - 1e-9 else "Constant"
    elif _LIN.search(q):
        rts, s = "Constant", 1.0
    else:
        return Result("unparsed", note="no Cobb-Douglas or linear form")
    if not any(re.search(r"increasing|decreasing|constant", o, re.I) for o in options):
        return Result("unparsed", note="options not in 'Returns to scale are X' form")
    norm = lambda o: re.sub(r"\s+", " ", o.strip().lower().rstrip("."))
    res = text_result(f"Returns to scale are {rts}", options, norm=norm)
    res.note = f"degree of homogeneity {s:g}"
    return res

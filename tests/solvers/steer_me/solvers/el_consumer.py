"""Consumer-theory elements: deriving_hicksian_demand, deriving_labor_supply, marginal_utility, mrs_utility,
diminishing_marginal_utility, compute_expected_utility.

Every solver re-derives the answer from the question text with textbook formulas.  Function-valued options
(e.g. '(0.44x^-0.56y^0.45, 0.45x^0.44y^-0.55)', '208.77w - 905.33/3.88w', '32.71(p_1^(-0.75) * p_2^(0.05))')
are parsed into small algebraic objects and compared numerically at a few fixed points (or coefficient by
coefficient), with the framework's 2 % relative tolerance absorbing 2-dp rounding of the shown parameters.

Small polynomial language used for utility functions and options: a *term* is `c * x^ex * y^ey`, a
*poly* is a list of terms, and a *ratio* is (numerator poly, denominator poly).
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

from .common import NUM, Result, close, match_result, nums, numeric_result, register, to_float, vector_result

REPO = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------------------- polynomials
_UNUM = r"\d+(?:\.\d+)?"                                   # unsigned number, no trailing '.'
_EXP = rf"\^\(?\s*-?{_UNUM}\s*\)?"                         # ^2, ^-0.56, ^(0.45), ^(-0.5)
_TERM = rf"(?:{_UNUM}\s*\*?\s*)?(?:[xy](?:{_EXP})?\s*\*?\s*)*[xy](?:{_EXP})?|{_UNUM}"
_POLY_RE = re.compile(rf"(?<![\w.^])(?:{_TERM})(?:\s*[+-]\s*(?:{_TERM}))*(?![\w^])")
_FREE_NUM = re.compile(rf"(?<![\w.^_])({_UNUM})(?![\w^])")   # a number not glued to a letter / exponent
_FACTOR_RE = re.compile(rf"([xy])(?:\^\(?\s*(-?{_UNUM})\s*\)?)?")


def parse_poly(s: str) -> list[tuple[float, float, float]] | None:
    """'4.49x + 17.04y', '0.44x^-0.56y^0.45', '24.22y^2 + 17.04x' -> [(c, ex, ey), ...]; None if not a poly."""
    s = s.strip()
    if not s:
        return None
    terms = []
    # split on binary +/- (a '-' directly after '^' is an exponent sign, not a separator)
    parts = re.split(r"(?<!\^)(?<!\^\()\s*([+-])\s*", s)
    sign = 1.0
    if parts and parts[0] == "":
        parts = parts[1:]
    for p in parts:
        if p in "+-":
            sign = -1.0 if p == "-" else 1.0
            continue
        m = re.fullmatch(rf"({_UNUM})?\s*\*?\s*((?:[xy](?:{_EXP})?\s*\*?\s*)*)", p.strip())
        if not m or (m.group(1) is None and not m.group(2)):
            return None
        c = float(m.group(1)) if m.group(1) else 1.0
        ex = ey = 0.0
        for v, e in _FACTOR_RE.findall(m.group(2)):
            e = float(e) if e else 1.0
            if v == "x":
                ex += e
            else:
                ey += e
        terms.append((sign * c, ex, ey))
        sign = 1.0
    return terms or None


def d_poly(poly, var: str):
    """Partial derivative of a poly with respect to 'x' or 'y'."""
    out = []
    for c, ex, ey in poly:
        e = ex if var == "x" else ey
        if e == 0:
            continue
        out.append((c * e, ex - 1, ey) if var == "x" else (c * e, ex, ey - 1))
    return out


def ev_poly(poly, x: float, y: float) -> float:
    return sum(c * x ** ex * y ** ey for c, ex, ey in poly)


# Evaluation points for comparing functions (x, y > 0; exponents may be negative or fractional).
_PTS = [(0.7, 1.9), (1.6, 0.8), (2.3, 1.3), (1.1, 2.6)]


def same_fn(f, g, pts=_PTS, rel=0.02, abs_=1e-9) -> bool:
    """True if callables f, g agree (relative tol) at every evaluation point."""
    for x, y in pts:
        a, b = f(x, y), g(x, y)
        if not (math.isfinite(a) and math.isfinite(b)):
            return False
        if abs(a - b) > max(abs_, rel * max(abs(a), abs(b))):
            return False
    return True


def _strip_parens(s: str) -> str:
    s = s.strip()
    return s[1:-1].strip() if s.startswith("(") and s.endswith(")") else s


def _utility_poly(q: str):
    """The utility function in the text: after 'U(x, y) =' / 'U =' if present, else the first poly-like span
    that contains a variable and a number/exponent (e.g. DMU's 'utility function x^1.1y^0.27, where ...')."""
    m = re.search(r"U\s*(?:\(\s*x\s*,\s*y\s*\))?\s*=\s*", q)
    spans = []
    if m:
        pm = _POLY_RE.match(q, m.end())
        if pm:
            spans.append(pm.group(0))
    for pm in _POLY_RE.finditer(q):
        g = pm.group(0)
        if re.search(r"[xy]", g) and re.search(r"\d|\^", g):
            spans.append(g)
    for sp in spans:
        p = parse_poly(sp)
        if p:
            return p, sp
    return None, None


# ---------------------------------------------------------------------------------------- marginal utility
@register("marginal_utility")
def marginal_utility(q, options, qtype):
    """MU vector (dU/dx, dU/dy).

    Cobb-Douglas x^a y^b -> (a x^(a-1) y^b, b x^a y^(b-1)); linear ax + by -> (a, b);
    'non-linear' c1 x^m + k xy + c2 y^n -> (m c1 x^(m-1) + k y, n c2 y^(n-1) + k x)  (note: for m = 1 the
    first component is the constant c1 + k y, NOT 'c1 x + k y').
    Leontief min(ax, by): MU = (a, 0) where ax < by, (0, b) where ax > by, undefined on the kink ax = by.
    The pair (a, b) is never the MU vector, so no option can be right; scored as zero-correct.
    """
    lm = re.search(rf"min\(\s*({NUM})\s*\*?\s*x\s*,\s*({NUM})\s*\*?\s*y\s*\)", q)
    if lm:
        a, b = float(lm.group(1)), float(lm.group(2))
        return Result("ok", f"piecewise: ({a}, 0) if {a}x<{b}y, (0, {b}) if {a}x>{b}y, undefined on kink",
                      [False] * len(options), None, note="Leontief MU is piecewise; (a, b) is never correct")
    U, _ = _utility_poly(q)
    if not U:
        return Result("unparsed", note="utility function not found")
    mux, muy = d_poly(U, "x"), d_poly(U, "y")
    rel = _fn_rel(U)

    def same(o):
        parts = [p.strip() for p in _strip_parens(o).split(",")]
        if len(parts) != 2:
            return None
        px, py = parse_poly(parts[0]), parse_poly(parts[1])
        if px is None or py is None:
            return None
        return (same_fn(lambda x, y: ev_poly(px, x, y), lambda x, y: ev_poly(mux, x, y), rel=rel)
                and same_fn(lambda x, y: ev_poly(py, x, y), lambda x, y: ev_poly(muy, x, y), rel=rel))

    return match_result(f"({_fmt(mux)}, {_fmt(muy)})", options, same)


def _fn_rel(U) -> float:
    """Comparison tolerance for derivative options.  Linear: the options repeat the shown 2-dp coefficients
    exactly, so only option rounding (abs 0.011 ~ rel 0.1 %) is allowed.  Cobb-Douglas (one term): coefficients
    and exponents are exact copies up to 2-dp rounding (<= 0.005 in an exponent ~ 0.5 % at our points) -> 1 %.
    Polynomials: derived coefficients (n*c) come from unrounded c -> the framework's 2 %."""
    if all(ex + ey <= 1 and ex in (0, 1) and ey in (0, 1) and not (ex and ey) for _, ex, ey in U):
        return 0.001
    return 0.01 if len(U) == 1 else 0.02


def _fmt(poly) -> str:
    out = []
    for c, ex, ey in poly:
        t = f"{c:.4g}"
        if ex:
            t += "x" if ex == 1 else f"x^{ex:.4g}"
        if ey:
            t += "y" if ey == 1 else f"y^{ey:.4g}"
        out.append(t)
    return " + ".join(out) or "0"


# ---------------------------------------------------------------------------------------- MRS
@register("mrs_utility")
def mrs_utility(q, options, qtype):
    """MRS_xy = MU_x / MU_y.  Options show the pair (MU_x/MU_y, MU_y/MU_x); both components must match, in
    that order.  Cobb-Douglas x^a y^b -> (a y / (b x), b x / (a y)); linear ax + by -> (a/b, b/a);
    polynomial -> ratio of the partial derivatives (options written 'N/D' are read as (N)/(D)).
    Leontief: MRS is 0 / infinite off the kink and undefined on it -> the option 'Undefined'."""
    if re.search(r"min\(", q):
        return match_result("Undefined", options, lambda o: bool(re.search(r"undefined", o, re.I)))
    U, _ = _utility_poly(q)
    if not U:
        return Result("unparsed", note="utility function not found")
    mux, muy = d_poly(U, "x"), d_poly(U, "y")
    if all(ex == 0 and ey == 0 for c, ex, ey in mux + muy):     # linear: numeric options 'a/b, b/a'
        a, b = ev_poly(mux, 1, 1), ev_poly(muy, 1, 1)
        # the shown a, b are 2-dp rounded: accept anything inside the interval image of [a +- .005]/[b +- .005]
        r_lo, r_hi = (a - 0.005) / (b + 0.005), (a + 0.005) / max(b - 0.005, 1e-9)
        i_lo, i_hi = (b - 0.005) / (a + 0.005), (b + 0.005) / max(a - 0.005, 1e-9)
        r = vector_result((a / b, b / a), options)
        if r.status == "ok":
            vecs = [nums(o) for o in options]
            r.correct = [r_lo - 0.011 <= v[0] <= r_hi + 0.011 and i_lo - 0.011 <= v[1] <= i_hi + 0.011
                         for v in vecs]
        return r
    # CD: the option coefficients are the shown exponents verbatim, so the ratio is exact -> 0.4 %
    rel = 0.004 if len(U) == 1 else _fn_rel(U)
    f1 = lambda x, y: ev_poly(mux, x, y) / ev_poly(muy, x, y)  # noqa: E731
    f2 = lambda x, y: ev_poly(muy, x, y) / ev_poly(mux, x, y)  # noqa: E731

    def ratio(s):
        if s.count("/") != 1:
            return None
        # '(0.59y)/(0.11x)' (parenthesised numerator and denominator) or '0.59y/0.11x'
        n, d = (parse_poly(_strip_parens(t)) for t in s.split("/"))
        if n is None or d is None:
            return None
        return lambda x, y: ev_poly(n, x, y) / ev_poly(d, x, y)

    def same(o):
        parts = [p.strip() for p in _strip_parens(o).split(",")]
        if len(parts) != 2:
            return None
        r1, r2 = ratio(parts[0]), ratio(parts[1])
        if r1 is None or r2 is None:
            return None
        return same_fn(r1, f1, rel=rel) and same_fn(r2, f2, rel=rel)

    return match_result(f"({_fmt(mux)})/({_fmt(muy)}) and inverse", options, same)


# ---------------------------------------------------------------------------------------- diminishing MU
_DMU_OPT = re.compile(rf"marginal utility of ([xy]) (increases from|decreases from|remains constant at) "
                      rf"({NUM})(?: to ({NUM}))?", re.I)


@register("diminishing_marginal_utility")
def diminishing_marginal_utility(q, options, qtype):
    """One good is held fixed, the other rises from `start` to `end`; describe its marginal utility.

    MU of the varied good v is dU/dv evaluated at (start, fixed) and (end, fixed):
      linear a x + b y       -> MU_x = a: constant
      Cobb-Douglas x^a y^b   -> MU_x = a x^(a-1) y^b: decreasing iff a < 1 (increasing if a > 1, constant a = 1)
      quadratic a x + b y + c x^2 + d y^2 + e xy -> MU_x = a + 2c x + e y: increasing iff c > 0
    An option is correct iff its direction AND its numbers match (MU(start) and, if given, MU(end)).
    The note records the direction-only verdict ('dir_ok=<i,...>') for a lenient re-scoring.
    """
    U, span = _utility_poly(q)
    if not U:
        return Result("unparsed", note="utility function not found")
    rest = q.replace(span, " ", 1)
    rng_m = re.search(rf"from ({_UNUM})(?: [a-z]+)? to ({_UNUM})", rest)
    if not rng_m:
        return Result("unparsed", note="start / end not found")
    fixed = _first(rf"(?:fix\w*|constant|held|kept|remains?|stays?|set|maintain\w*|keep\w*|only use)\D{{0,40}}?"
                   rf"(?<![\w.^])({_UNUM})(?![\w^])", rest)
    if fixed is None:     # fallback: the single free number that is not start/end
        free = [float(x) for x in _FREE_NUM.findall(rest[:rng_m.start()] + " " + rest[rng_m.end():])]
        fixed = free[0] if len(free) == 1 else None
    if fixed is None:
        return Result("unparsed", note="fixed quantity not found")
    start, end = float(rng_m.group(1)), float(rng_m.group(2))
    var = "x" if qtype.startswith("y_fixed") else "y" if qtype.startswith("x_fixed") else None
    ovars = {m.group(1).lower() for o in options for m in [_DMU_OPT.search(o)] if m}
    if var is None:
        var = ovars.pop() if len(ovars) == 1 else None
    if var is None:
        return Result("unparsed", note="varied good unknown")
    mu = d_poly(U, var)
    at = (lambda v: ev_poly(mu, v, fixed)) if var == "x" else (lambda v: ev_poly(mu, fixed, v))
    m0, m1 = at(start), at(end)
    if not (math.isfinite(m0) and math.isfinite(m1)):
        return Result("undefined", note="MU not finite")
    diff = m1 - m0
    direction = "constant" if abs(diff) <= 1e-9 * max(1.0, abs(m0)) else ("increases" if diff > 0 else "decreases")
    corr, dir_ok = [], []
    for o in options:
        m = _DMU_OPT.search(o)
        if not m:
            return Result("unparsed", note="option not recognised")
        d = {"i": "increases", "d": "decreases", "r": "constant"}[m.group(2)[0].lower()]
        a = float(m.group(3))
        b = float(m.group(4)) if m.group(4) else None
        vals_ok = close(a, m0) and (b is None or close(b, m1))
        corr.append(d == direction and m.group(1).lower() == var and vals_ok)
        dir_ok.append(d == direction)
    note = f"MU_{var}: {m0:.4g} -> {m1:.4g}; dir_ok={','.join(str(i) for i, v in enumerate(dir_ok) if v)}"
    return Result("ok", f"{direction} {m0:.4g} -> {m1:.4g}", corr, corr.index(True) if any(corr) else
                  (dir_ok.index(True) if any(dir_ok) else None), note)


def _first(pattern, s):
    m = re.search(pattern, s, re.I | re.S)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------------------- Hicksian demand
# Parameters are rendered with a decimal point ('5.0', '491.0'), so bare integers ('Asset 1', 'good 2') are labels.
_DEC_NUM = re.compile(r"(?<![\w.^_])(\d+\.\d+)(?![\w^])")
_LEVEL_KW = re.compile(r"(level|at least|minimum|utility|satisfaction|threshold|standard|target|reach|achieve|"
                       r"maintain|benefit)\W*(?:[\w’']+\W+){0,6}$", re.I)


def _hclose(opt: str, val: float, exp: float) -> bool:
    """Cobb-Douglas inputs (exponents, prices, u) are exact 2-dp numbers, so the right option is `exp` rounded
    to the precision it is printed with: 2 dp (half-unit 0.005), or 3 significant digits in '1.66e+34' form.
    This separates the filler's near-duplicate distractors (e.g. 423.10 vs 423.91)."""
    if "e" in opt.lower():
        return abs(val - exp) <= 0.006 * abs(exp)
    return abs(val - exp) <= 0.006 + 1e-4 * abs(exp)


@register("deriving_hicksian_demand")
def deriving_hicksian_demand(q, options, qtype):
    """Compensated demand for good 1: min p.x s.t. U(x) >= u.

    Cobb-Douglas prod x_j^a_j (s = sum a_j): FOC p_j x_j / a_j = k for all j, so
        h_1 = (a_1/p_1) * (u / prod_j (a_j/p_j)^a_j)^(1/s)
            = [a_1 (u / prod_j a_j^a_j)^(1/s)] * p_1^(a_1/s - 1) * prod_{j>1} p_j^(a_j/s)
    Numeric prices -> a number (tolerance: printed precision, see _hclose);
    symbolic prices (p_x, p_y / p_1..p_3) -> options 'C(p_1^(e1) * ...)' compared on C and every exponent.
    Linear sum a_j x_j: buy only the good with the highest a_j/p_j; h_1 = u/a_1 if that is good 1, else 0.
    The shown a_j are rounded to 2 dp (true a_j in [a_j -/+ 0.005]); 'undefined' if that rounding can flip
    which good is cheapest per util, and any option inside [u/(a_1+.005), u/(a_1-.005)] counts as correct.
    Assumes the target is met with equality (U = u); the question always asks for the first good (good_1).
    """
    lin = list(re.finditer(rf"(?<![\^\d.])({_UNUM})\s*\*\s*x(\d)(?!\s*\^)", q))
    cd = list(re.finditer(rf"(?<![\w])(x\d?|y|z)\^\(?\s*({_UNUM})\s*\)?", q))
    if len(lin) >= 2:
        coef = [float(m.group(1)) for m in sorted(lin, key=lambda m: m.group(2))]
        span = (lin[0].start(), lin[-1].end())
        c0 = re.match(rf"\s*\+\s*{_UNUM}(?!\s*\*)", q[span[1]:])     # trailing constant, if any
        span = (span[0], span[1] + (c0.end() if c0 else 0))
        kind = "linear"
    elif len(cd) >= 2:
        coef = [float(m.group(2)) for m in cd]
        span, kind = (cd[0].start(), cd[-1].end()), "cd"
    else:
        return Result("unparsed", note="utility function not found")
    n = len(coef)
    text = q[:span[0]] + " " * (span[1] - span[0]) + q[span[1]:]
    nums_pos = [(m.start(), float(m.group(1))) for m in _DEC_NUM.finditer(text)]
    symbolic = bool(re.search(r"\bp_(?:x|y|1|2|3)\b", text))
    lvl = [i for i, (pos, _) in enumerate(nums_pos) if _LEVEL_KW.search(text[max(0, pos - 80):pos])]
    if len(nums_pos) == (1 if symbolic else n + 1) and not lvl:
        lvl = [max(range(len(nums_pos)), key=lambda i: nums_pos[i][1])]      # fallback: the big number
    if not lvl:
        return Result("unparsed", note="utility level not found")
    li = lvl[-1] if symbolic or len(lvl) == 1 else next((i for i in lvl if nums_pos[i][1] >= 10), lvl[-1])
    u = nums_pos[li][1]
    prices = [v for i, (_, v) in enumerate(nums_pos) if i != li]
    s = sum(coef)
    if kind == "linear":
        if symbolic or len(prices) != n:
            return Result("unparsed", note="linear case needs n numeric prices")
        lo = [(c - 0.005) / pi for c, pi in zip(coef, prices)]
        hi = [(c + 0.005) / pi for c, pi in zip(coef, prices)]
        note = f"a/p={['%.3g' % (c / pi) for c, pi in zip(coef, prices)]}"
        if n > 1 and lo[0] > max(hi[1:]):
            h, band = u / coef[0], (u / (coef[0] + 0.005), u / max(coef[0] - 0.005, 1e-9))
        elif n > 1 and hi[0] < max(lo[1:]):
            h, band = 0.0, (0.0, 0.0)
        else:
            return Result("undefined", note="2-dp rounding of the coefficients decides which good is bought; " + note)
        vals = [to_float(o) for o in options]
        if any(v is None for v in vals):
            return Result("unparsed", note="non-numeric option")
        corr = [band[0] - 0.011 <= v <= band[1] + 0.011 for v in vals]
        closest = min(range(len(vals)), key=lambda i: abs(vals[i] - h))
        return Result("ok", round(h, 6), corr, closest, note)
    if len(prices) != (0 if symbolic else n):
        return Result("unparsed", note=f"expected {n} prices, found {len(prices)}")
    if coef[0] == 0:
        return (Result("undefined", note="a_1 = 0") if symbolic else numeric_result(0.0, options, note="a_1 = 0"))
    if symbolic:
        C = coef[0] * (u / math.prod(a ** a for a in coef)) ** (1 / s)
        exps = [coef[j] / s - (1 if j == 0 else 0) for j in range(n)]

        def same(o):
            m = re.fullmatch(rf"\s*({NUM})\s*\*?\s*\((.*)\)\s*", o)
            if not m:
                return None
            es = [float(e) for e in re.findall(rf"p_\w+\^\(\s*({NUM})\s*\)", m.group(2))]
            if len(es) != n:
                return None
            return _hclose(m.group(1), float(m.group(1)), C) and all(abs(a - b) <= 0.011 for a, b in zip(es, exps))
        return match_result(f"{C:.6g}*prod p^{[round(e, 3) for e in exps]}", options, same)
    h = (coef[0] / prices[0]) * (u / math.prod((a / pi) ** a for a, pi in zip(coef, prices))) ** (1 / s)
    r = numeric_result(h, options)
    if r.status == "ok":
        r.correct = [_hclose(o, to_float(o), h) for o in options]
    return r


# ---------------------------------------------------------------------------------------- labor supply
T_ENDOW = 24.0   # ASSUMPTION: the time endowment is never stated; 24 hours (per day) is used.


@register("deriving_labor_supply")
def deriving_labor_supply(q, options, qtype):
    """max C^a L^b  s.t.  C = w h + v,  L = T - h   (T = 24, not stated in the text).

    Cobb-Douglas share rule: L* = b/(a+b) * (wT + v)/w, so
        h*(w) = T - L* = (a T w - b v) / ((a + b) w)       (interior; h* = 0 if a T w < b v).
    'hours' types: a number.  'function' types: options 'Aw - B/Dw' are read as (A w - B)/(D w) and
    compared through A/D = aT/(a+b) and B/D = b v/(a+b).
    """
    m = re.search(rf"C\^\(?\s*({_UNUM})\s*\)?\s*\*?\s*L\^\(?\s*({_UNUM})\s*\)?", q)
    if not m:
        return Result("unparsed", note="utility C^a L^b not found")
    a, b = float(m.group(1)), float(m.group(2))
    rest = q[:m.start()] + " " + q[m.end():]
    rest = re.sub(r"wh\s*\+\s*v where w is the (?:wage rate|hourly wage in dollars), h is the number of hours worked"
                  r"(?: per day)?, and v is non-labor income(?: in dollars per day)?"
                  r"(?:, with 24 hours per day to divide between work h and leisure L = 24 - h)?"  # me-fillers
                  r"(?:, and the wage is high enough that the worker chooses to work some hours \(h > 0\))?",
                  " BUDGET ", rest)                                   # the fixed budget sentence names w, v
    nums_pos = [(mm.start(), float(mm.group(1))) for mm in _DEC_NUM.finditer(rest)]
    fn_type = "function" in qtype
    if fn_type:
        if len(nums_pos) != 1:
            return Result("unparsed", note=f"expected only the non-labor income, found {len(nums_pos)} numbers")
        v = nums_pos[0][1]
        k1, k2 = a * T_ENDOW / (a + b), b * v / (a + b)

        def same(o):
            # me-fillers: also accept the parenthesised form "(Aw - B)/(Dw)"
            mm = re.fullmatch(rf"\s*\(?\s*({NUM})\s*w\s*-\s*({NUM})\s*\)?\s*/\s*\(?\s*({NUM})\s*w\s*\)?\s*", o)
            if not mm:
                return None
            A, B, D = map(float, mm.groups())
            return D != 0 and close(A / D, k1) and close(B / D, k2)
        return match_result(f"({k1:.4g}w - {k2:.4g})/w  [T=24]", options, same)
    if len(nums_pos) != 2:
        return Result("unparsed", note=f"expected wage and non-labor income, found {len(nums_pos)} numbers")
    # w: the number closest after a wage keyword; v: the other one
    kw = [k.end() for k in re.finditer(r"wage|earn|hourly|per hour|salary|compensation|\bpay|\brate|\bfee", rest, re.I)]
    # "you earn 82.59 from investments" names the non-labor income, not the wage
    kw = [e for e in kw if not re.match(r"\s*\$?[\d.]+\s+(?:from|in|through) (?:investment|interest|dividend|saving|rent)",
                                         rest[e:], re.I)]
    # the first number in the text shortly after a wage keyword ("His hourly wage is 37.34, and he earns 96.97 ...")
    close_after = [(p, x) for p, x in nums_pos if any(0 <= p - e <= 15 for e in kw)]
    w = min(close_after)[1] if close_after else _nearest_after(nums_pos, kw)
    if w is None:
        return Result("unparsed", note="wage not identified")
    v = nums_pos[1][1] if nums_pos[0][1] == w else nums_pos[0][1]
    h = (a * T_ENDOW * w - b * v) / ((a + b) * w)
    note = "T=24 assumed"
    if h < 0:
        h, note = 0.0, "corner: interior h<0 -> h=0 (T=24 assumed)"
    return numeric_result(h, options, note=note)


def _nearest_after(nums_pos, kw_ends):
    """Value of the number with the smallest positive distance after any keyword occurrence."""
    best = None
    for e in kw_ends:
        for pos, val in nums_pos:
            if pos >= e and (best is None or pos - e < best[0]):
                best = (pos - e, val)
    return best[1] if best else None


# ---------------------------------------------------------------------------------------- expected value
@lru_cache(maxsize=None)
def _eu_templates():
    """Template regexes for compute_expected_utility: the placeholders name the quantities in the text
    (win_amount, lose_chance, num_days, ...).  Only the template *text* is used, never the filler."""
    from autosteer.paths import templates_dir
    from autosteer.datasets import Dataset
    try:
        df = Dataset.load(str(templates_dir() / "expected_value_counts_aug2024")).questions
    except Exception:  # noqa: BLE001
        return []
    out = []
    for text, tags in zip(df["question_text"], df["tags"]):
        pat, seen, pos = "", set(), 0
        text = re.sub(r"\s+", " ", text.strip())
        for m in re.finditer(r"\{(\w+)\}", text):
            pat += re.escape(text[pos:m.start()])
            name = m.group(1)
            pat += f"(?P={name})" if name in seen else f"(?P<{name}>.+?)"
            seen.add(name)
            pos = m.end()
        pat += re.escape(text[pos:])
        out.append((re.compile(pat, re.S), str(tags)))
    return out


def _val(s):
    s = s.strip().rstrip(".")
    pct = s.endswith("%")
    x = to_float(s)
    return None if x is None else (x / 100 if pct else x)


_BAD = (r"not (?:be )?(?:ski|hik|surviv|saved|recover|go)|confin|snowed|indoor|inside|off the slopes|stuck|miss|unable|"
        r"cannot|can't|prevent|lost|bad weather|blizzard|storm|fog|rain|die|death|dead|fatal|not survive|harm|"
        r"side effect|complication|adverse|fail|cripp|not (?:be )?cured|lose their|succumb")


# Inputs are exact (integers and 2-dp percentages of the same draw), options are 2-dp: tight tolerance.
EU_REL, EU_ABS = 0.002, 0.006


@register("expected_value_counts_aug2024")  # the Aug-2024 variant, renamed 2026-10-02
def compute_expected_utility(q, options, qtype):
    """Two-outcome expectations (risk-neutral, i.e. expected value):

    vacation: E[#good days] = n p_good, E[#bad days] = n p_bad (the question sentence decides which);
    medical:  E[#saved] = N p_save, E[#harmed / not saved] = N p_not_save;
    gambling: per game EV = p_win W - p_lose L (L = the amount lost in the losing outcome, if stated);
              'expected win / value' -> n * EV, 'expected loss' -> -n * EV  (NET reading, textbook EV).
    The filler instead uses the GROSS component (p_win W n or p_lose L n); the note records whether the gross
    reading matches the marked option ('gross_idx=i') so both readings can be tallied.
    Values are extracted by matching the text to its template (placeholders give the quantities' roles);
    the asked quantity is read from the final question sentence, not from the template tag.
    """
    qn = re.sub(r"\s+", " ", q.strip())
    hit = None
    for rx, tags in _eu_templates():
        m = rx.fullmatch(qn)
        if m:
            hit = (m.groupdict(), tags)
            break
    if not hit:
        return Result("unparsed", note="no template matches")
    g, tags = hit
    vals = {k: _val(v) for k, v in g.items()}
    if any(v is None for v in vals.values()):
        return Result("unparsed", note="placeholder value not numeric")
    last = re.split(r"(?<=[.!])\s+", qn)[-1].lower()
    # the interrogative clause ('how many days can you expect to spend skiing?'), not preceding conditions
    wh = list(re.finditer(r"\b(how many|how much|what|which|calculate|determine|compute|estimate|find)\b", last))
    ask = last[wh[-1].start():] if wh else last
    # me-fillers: "without <bad thing>" (any length up to the clause end) excludes the bad outcome, e.g.
    # "without suffering from severe side effects", "without the inconvenience of fog", "without dealing
    # with storms"; "not be able to hike" asks for the bad outcome.
    ask_ = re.sub(r"without [^,.?;]*", lambda m: " " if re.search(_BAD, m.group(0)) else m.group(0), ask)
    bad = bool(re.search(_BAD, ask_) or re.search(r"without (?:ski|hik|go)|not be able to", ask))
    note = f"tags={tags}"
    if "num_days" in vals and "good_chance" in vals:
        p = vals.get("bad_chance", 1 - vals["good_chance"]) if bad else vals["good_chance"]
        exp = vals["num_days"] * p
        return _num_with_note(exp, options, note + f"; asks {'bad' if bad else 'good'} days")
    if "num_patients" in vals and "save_chance" in vals:
        p = vals.get("not_save_chance", 1 - vals["save_chance"]) if bad else vals["save_chance"]
        return _num_with_note(vals["num_patients"] * p, options, note + f"; asks {'harmed' if bad else 'saved'}")
    if "win_amount" in vals and ("win_chance" in vals or "lose_chance" in vals):
        pw = vals.get("win_chance", 1 - vals.get("lose_chance", 0))
        pl = vals.get("lose_chance", 1 - pw)
        W, L = vals["win_amount"], vals.get("lose_amount", 0.0)
        n = vals.get("num_games", 1.0)
        # 'lose' / 'loss' / 'cost' asks for the expected loss, unless the clause asks for a signed result
        # ('win (or lose)', 'gain or lose', 'expected value of the game') -> expected (net) winnings
        signed = re.search(r"(win|won|gain)\w* \(?or (lose|lost|loss)|value of (the|this) game", ask)
        asks_loss = bool(re.search(r"\blos[et]|\bloss|\bcost", ask)) and not signed
        ev = pw * W - pl * L
        net = -n * ev if asks_loss else n * ev
        gross = n * pl * L if asks_loss else n * pw * W
        r = numeric_result(net, options, rel=EU_REL, abs_=EU_ABS, note=note)
        gr = numeric_result(gross, options, rel=EU_REL, abs_=EU_ABS)
        if r.status == "ok":
            gi = [i for i, c in enumerate(gr.correct or []) if c]
            r.note = (f"{note}; asks {'loss' if asks_loss else 'win'}; net={net:.4g} gross={gross:.4g}; "
                      f"gross_idx={','.join(map(str, gi))}")
        return r
    return Result("unparsed", note=f"template placeholders not handled: {sorted(vals)}")


def _num_with_note(exp, options, note):
    r = numeric_result(exp, options, rel=EU_REL, abs_=EU_ABS)
    r.note = note
    return r

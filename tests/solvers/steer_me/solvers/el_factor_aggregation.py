"""cost_minimization_problem, dynamic_profit_max, aggregation_of_{capital_demand,labor_supply,offer_curve}.

All answers are re-derived from the question text with textbook formulas (see each solver's docstring).
Displayed parameters are rounded to 2 dp; where the generator used unrounded values (dynamic_profit_max's
discount factor) the tolerance is a band obtained by re-solving at +-0.005.  Aggregation coefficients are
exact, so those comparisons are tight.
"""
from __future__ import annotations

import re

from .common import ABS_TOL, NUM, Result, close, nums, numeric_result, register, to_float

# ============================================================================ cost minimisation
_L_TAG = re.compile(r"\(L\)|\blabou?r\b|\bworkers?\b|\bstaff\b|\bemploy|\bcoordinator|\bpersonnel|\bwage", re.I)
_K_TAG = re.compile(r"\(K\)|\bcapital\b|\bequipment\b|\bmachine|\btechnolog|\binfrastructure|\brent", re.I)


_OUT_WORDS = re.compile(r"least|minimum|target|goal|produc|output|reach|meet|exceed|accommodat|deliver|handle|"
                       r"achieve|process|require|need|level of|quota|demand", re.I)
_COST_WORDS = re.compile(r"cost|price|priced|expens|wage|rent|pay|incur|charg|spend|\$", re.I)
_CLAUSE_END = re.compile(r",|;|\.\s|\band\b|\bwhile\b|\bwhereas\b")


def _after_label(after: str) -> str:
    """Text of a trailing 'per <thing>' phrase right after a number (at most 3 words, stops at and/with/,)."""
    m = re.match(r"\s*(?:per|for each|for every|in|on|for)\s+", after)
    if not m:
        return ""
    words = []
    for w in re.findall(r"\(?[\w'-]+\)?|[,;.]", after[m.end():])[:3]:
        if w in (",", ";", ".") or w.lower() in ("and", "with", "while", "whereas", "but", "against", "versus",
                                                 "vs", "compared", "plus"):
            break
        words.append(w)
    return " ".join(words)


def _strip_leading_per(before: str) -> str:
    """Drop the previous number's trailing 'per <thing>' phrase from a preceding window."""
    lab = _after_label(before)
    if not lab:
        return before
    last = lab.split()[-1]
    return before[before.index(last) + len(last):]


def _factor_label(before: str, after: str) -> str | None:
    """'L' / 'K' if the words attached to a cost figure name exactly one factor.

    Looks first at a trailing 'per <thing>' phrase, then at the text since the previous number;
    explicit '(L)' / '(K)' beats keyword matches.
    """
    for window in (_after_label(after), _strip_leading_per(before)):
        has_l, has_k = "(L)" in window, "(K)" in window
        if has_l != has_k:
            return "L" if has_l else "K"
        has_l, has_k = bool(_L_TAG.search(window)), bool(_K_TAG.search(window))
        if has_l != has_k:
            return "L" if has_l else "K"
    return None


# All cost-minimisation inputs are shown exactly (rounded before use); the slack only absorbs the 2-dp
# rounding of the options and the generator's numerical optimiser tolerance.
CM_REL = 0.001


def _cd_parse(q: str):
    """Find 'A L^a K^b' (either factor order). Returns (A, a, b, span) or None."""
    m = re.search(rf"({NUM})?\s*\*?\s*L\^\(?({NUM})\)?\s*\*?\s*K\^\(?({NUM})\)?", q)
    if m:
        return (float(m.group(1)) if m.group(1) else 1.0), float(m.group(2)), float(m.group(3)), m.span()
    m = re.search(rf"({NUM})?\s*\*?\s*K\^\(?({NUM})\)?\s*\*?\s*L\^\(?({NUM})\)?", q)
    if m:
        return (float(m.group(1)) if m.group(1) else 1.0), float(m.group(3)), float(m.group(2)), m.span()
    return None


@register("cost_minimization_problem")
def cost_minimization_problem(q, options, qtype):
    """min wL + rK  s.t.  A L^a K^b >= y  (Cobb-Douglas, a, b > 0).

    The isoquant of a Cobb-Douglas function is strictly convex for any a, b > 0 (also with increasing
    returns), so the tangency condition MRTS = w/r gives the unique cost minimum:
        K/L = (b w) / (a r),   L* = (y / (A (K/L)^b))^(1/(a+b)),   K* = (K/L) L*,   C* = w L* + r K*.
    compute_labor -> L*, compute_capital -> K*, cost_minimizing -> C*.
    Numbers: after removing the production function the text must contain exactly 3 numbers, in the order
    output, cost, cost; each cost is assigned to L or K from the words just before it (default: labour first).
    """
    cd = _cd_parse(q)
    if not cd:
        return Result("unparsed", note="no Cobb-Douglas L^a K^b")
    A, a, b, (s, e) = cd
    rest = q[:s] + " FUNC " + q[e:]
    ms = list(re.finditer(NUM, rest))
    if len(ms) != 3:
        return Result("unparsed", note=f"expected 3 numbers besides p_func, got {len(ms)}")
    prev_end = [0, ms[0].end(), ms[1].end()]
    # preceding text of each number, limited to its own sentence
    befores = [re.split(r"[.?!]\s", _strip_leading_per(rest[prev_end[i]:m.start()]))[-1] for i, m in enumerate(ms)]
    afters = [rest[m.end():m.end() + 60] for m in ms]
    # output = the number introduced by output words and not by cost words (default: the first one)
    is_out = [bool(_OUT_WORDS.search(b)) and not _COST_WORDS.search(b) and not _COST_WORDS.search(_after_label(a))
              for b, a in zip(befores, afters)]
    iy = is_out.index(True) if sum(is_out) == 1 else 0
    note = "" if iy == 0 else "output stated after the costs"
    ic = [i for i in range(3) if i != iy]
    y = float(ms[iy].group())
    c1, c2 = (float(ms[i].group()) for i in ic)
    lab1, lab2 = (_factor_label(rest[prev_end[i]:ms[i].start()], afters[i]) for i in ic)
    if (lab1, lab2) in {("K", "L"), ("K", None), (None, "L")}:
        w, r = c2, c1
        note = (note + "; " if note else "") + "costs stated in K, L order"
    else:
        w, r = c1, c2
        if lab1 == lab2 and lab1 is not None:
            note = (note + "; " if note else "") + "both costs look like the same factor; assumed L, K order"
    if a == 0 or b == 0:  # one input is useless -> buy none of it
        if min(A, y, w, r) <= 0 or a == b == 0:
            return Result("undefined", note="degenerate production function")
        L, K = (0.0, (y / A) ** (1 / b)) if a == 0 else ((y / A) ** (1 / a), 0.0)
        target = {"compute_labor": L, "compute_capital": K, "cost_minimizing": w * L + r * K}.get(qtype)
        return numeric_result(target, options, rel=CM_REL, note=(note + "; " if note else "") + "zero exponent shown")
    if min(a, b, w, r, y, A) <= 0:
        return Result("undefined", note="non-positive parameter")
    ratio = (b * w) / (a * r)
    L = (y / (A * ratio ** b)) ** (1.0 / (a + b))
    K = ratio * L
    target = {"compute_labor": L, "compute_capital": K, "cost_minimizing": w * L + r * K}.get(qtype)
    if target is None:
        return Result("unparsed", note=f"unknown type {qtype}")
    return numeric_result(target, options, rel=CM_REL, note=note)


# ============================================================================ dynamic profit max
def _solve_K2(K1: float, coef: float, alpha: float) -> float | None:
    """argmax_{K2 >= 0}  coef * K2^alpha - (K2 - K1)^2   (coef = beta E[p] A > 0).

    alpha < 1: strictly concave, unique root of 2 (K2 - K1) = coef alpha K2^(alpha-1) (bisection).
    alpha = 1: K2 = K1 + coef / 2.   1 < alpha < 2: bounded but maybe not concave -> grid + refine.
    alpha >= 2: unbounded (alpha > 2, or alpha = 2 with coef >= 1) -> None.
    """
    if alpha < 1:
        g = lambda k: 2 * (k - K1) - coef * alpha * k ** (alpha - 1)
        lo, hi = 1e-12, max(K1, 1.0)
        while g(hi) < 0:
            hi *= 2
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
        return 0.5 * (lo + hi)
    if alpha == 1:
        return K1 + coef / 2
    if alpha >= 2:
        return None
    h = lambda k: coef * k ** alpha - (k - K1) ** 2
    U = 10 * (K1 + 1 + (coef * alpha) ** (1 / (2 - alpha)))
    n = 20000
    k0 = max(range(n + 1), key=lambda i: h(U * i / n)) * U / n
    lo, hi = max(0.0, k0 - U / n), k0 + U / n
    for _ in range(200):  # golden-section refine
        a, b = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        lo, hi = (lo, b) if h(a) > h(b) else (a, hi)
    return 0.5 * (lo + hi)


@register("dynamic_profit_max")
def dynamic_profit_max(q, options, qtype):
    """Two-period capital choice.

    Profit = p1 f(K1) - (K1 - K2)^2 + beta * E[p2] * f(K2),  f(K) = A K^alpha (see _solve_K2 for alpha >= 1).
    (p1 f(K1) is sunk, so today's price is irrelevant.)  FOC: 2 (K2 - K1) = beta E[p2] A alpha K2^(alpha-1);
    for alpha < 1 the LHS rises and the RHS falls in K2, so the root is unique and K2 > K1 (alpha = 0: no
    investment; alpha = 1: K2 = K1 + beta E[p2] A / 2).  Answer = the increase K2 - K1
    ("how much should X increase capital" / "optimal capital investment"), since the options are on that
    scale.  E[p2] = sum p_i q_i over the stated distribution (probabilities as displayed, not renormalised).
    Assumption: the adjustment cost is paid today (undiscounted), only revenue tomorrow is discounted.
    """
    m = re.search(rf"({NUM})?\s*\*?\s*K\^\(?({NUM})\)?", q)
    if not m:
        return Result("unparsed", note="no A K^alpha")
    A = float(m.group(1)) if m.group(1) else 1.0
    alpha = float(m.group(2))
    # K_1 value: 'K_1 = 5.3', 'K_1, is equal to 5.3', 'K_1 being 5.3', '(K_1) of 5.3', 'K_1 and equals 5.3',
    # or no symbol at all: 'capital at day 1 is (denoted as) 5.3'
    mk = (re.search(rf"K_1(?![-\s]*K_2)\)?[^0-9.?!]{{0,30}}?({NUM})", q)
          or re.search(rf"(?:day|period) (?:1|one)\b[^0-9.?!]{{0,30}}?({NUM})", q, re.I)
          or re.search(rf"capital (?:of|is|stands at)\s*({NUM})", q, re.I))
    md = (re.search(rf"discount[^.]*?({NUM})", q, re.I) or re.search(rf"({NUM})\s*(?:discount )?factor", q, re.I)
          or re.search(rf"factor(?: of)?\s*({NUM})", q, re.I))
    if not (mk and md):
        return Result("unparsed", note="missing K_1 or discount factor")
    K1, beta = float(mk.group(1)), float(md.group(1))
    if not re.search(r"\(K_1\s*-\s*K_2\)\^2", q):
        return Result("unparsed", note="cost function not (K_1-K_2)^2")
    note = "K_1 displayed as 0.0" if K1 == 0 else ""
    if qtype == "uncertain_price":
        pairs = re.findall(rf"price\s*({NUM})\s*with probability\s*({NUM})", q, re.I)
        if len(pairs) < 2:
            return Result("unparsed", note="price distribution not found")
        Ep = sum(float(p) * float(pr) for p, pr in pairs)
        psum = sum(float(pr) for _, pr in pairs)
        if abs(psum - 1) > 0.02:
            note = (note + "; " if note else "") + f"probabilities sum to {psum:.3f}"
    elif qtype == "known_price":
        # remove p_func, K_1 value, discount and cost function; remaining numbers are p1 then p2
        rest = q
        for sp in sorted([m.span(), mk.span(1), md.span(1)], reverse=True):
            rest = rest[:sp[0]] + " X " + rest[sp[1]:]
        rest = re.sub(r"\(K_1\s*-\s*K_2\)\^2", " ", rest).replace("K_1", "").replace("K_2", "")
        rest = re.sub(r"\b(?:day|period|year|month|week|stage|time) \d\b", " ", rest, flags=re.I)
        ps = nums(rest)
        if len(ps) != 2:
            return Result("unparsed", note=f"expected 2 prices, got {ps}")
        Ep = ps[1]
    else:
        return Result("unparsed", note=f"unknown type {qtype}")
    if alpha == 0:  # output does not depend on capital: any increase only costs money
        return numeric_result(0.0, options, note=(note + "; " if note else "") + "alpha displayed as 0.0")
    if alpha < 0 or beta < 0:
        return Result("undefined", note=f"alpha={alpha}, beta={beta}")
    K2 = _solve_K2(K1, beta * Ep * A, alpha)
    if K2 is None:
        return Result("undefined", note=f"alpha={alpha}: profit unbounded in K2")
    # Rounding band: the generator draws the discount factor and the probabilities unrounded but shows them
    # to 2 dp (A, alpha, prices, K_1 are rounded before use), and the increase is roughly proportional to
    # beta.  So re-solve at beta +- 0.005 and E[p] +- 0.005 * sum(prices) and accept options in [min, max].
    dEp = 0.005 * sum(float(p) for p, _ in pairs) if qtype == "uncertain_price" else 0.0
    band = []
    for db in (-0.005, 0.005):
        for de in (-dEp, dEp):
            k = _solve_K2(K1, max(beta + db, 0) * (Ep + de) * A, alpha)
            if k is not None:
                band.append(k - K1)
    lo, hi = min(band + [K2 - K1]) - ABS_TOL, max(band + [K2 - K1]) + ABS_TOL
    vals = [to_float(o) for o in options]
    if any(v is None for v in vals):
        return Result("unparsed", note="non-numeric option")
    corr = [lo <= v <= hi for v in vals]
    closest = min(range(len(vals)), key=lambda i: abs(vals[i] - (K2 - K1)))
    if any(close(v, K2) for v in vals) and not close(K2, K2 - K1):
        note = (note + "; " if note else "") + "K2 level also among options"
    return Result("ok", round(K2 - K1, 6), corr, closest, note + f" band=[{lo:.3f},{hi:.3f}]")


# ============================================================================ aggregation
_LIN = re.compile(rf"({NUM})\s*([+-])\s*({NUM})\s*\*?\s*([PpWw])\b")


def _lin(s: str) -> tuple[float, float] | None:
    """'57928.86 - 1500.66P' -> (57928.86, -1500.66)."""
    m = _LIN.fullmatch(s.strip().rstrip("."))
    if not m:
        m = _LIN.search(s)
        if not m:
            return None
    a, sign, b = float(m.group(1)), m.group(2), float(m.group(3))
    return a, (b if sign == "+" else -b)


def _aggregation(q, options, qtype, func_type):
    """N_a units with q_a(P) = a1 + b1 P and N_b units with a2 + b2 P (horizontal summation):
        Q(P) = (N_a a1 + N_b a2) + (N_a b1 + N_b b2) P.
    Function types compare (intercept, slope); 'numerical' evaluates Q at the stated price/wage.
    Counts: if a total is stated, (N_a, N_b) are the two counts summing to it, in text order; otherwise
    N_a = the integer just before the first function and N_b = the one between the functions (or
    total - N_a when the text says 'the rest').
    Note flags an individual function that is negative at the stated price (no truncation at zero is
    applied, matching the plain linear reading; the note shows where max(0, .) would differ).
    """
    fs = list(_LIN.finditer(q))
    if len(fs) != 2:
        return Result("unparsed", note=f"expected 2 linear functions, got {len(fs)}")
    (a1, b1), (a2, b2) = [_lin(f.group()) for f in fs]
    rest = q
    for f in reversed(fs):
        rest = rest[:f.start()] + " FUNC " + rest[f.end():]
    ms = [x for x in re.finditer(NUM, rest)]
    val = lambda x: float(x.group().rstrip("."))  # '1333.' at a sentence end is an integer
    ints = [x for x in ms if re.fullmatch(r"\d+\.?", x.group())]
    fpos = [m.start() for m in re.finditer(r"FUNC", rest)]
    # 1) a stated total: two counts that add up to a third one -> (N_a, N_b) in text order
    # 2) no total: N_a = the count closest before the first function, N_b = the count between the functions,
    #    or total - N_a when the text says '..., whereas the rest follow ...'.
    note, Na = "", None
    iv = [(x.start(), int(val(x))) for x in ints]
    for (pt, t) in iv:
        pair = [(p1, x, p2, y) for (p1, x) in iv for (p2, y) in iv if p1 < p2 and pt not in (p1, p2) and x + y == t]
        if pair:
            p1, Na, p2, Nb = pair[0]
            used = {pt, p1, p2}
            break
    if Na is None:
        pre1 = [x for x in ints if x.start() < fpos[0]]
        mid = [x for x in ints if fpos[0] < x.start() < fpos[1]]
        if not pre1:
            return Result("unparsed", note="no count before the first function")
        Na = int(val(pre1[-1]))
        if mid:
            Nb = int(val(mid[-1]))
            used = {pre1[-1].start(), mid[-1].start()}
        elif len(pre1) >= 2 and int(val(pre1[-2])) > Na:
            Nb = int(val(pre1[-2])) - Na
            used = {pre1[-1].start(), pre1[-2].start()}
            note = "N_b = total - N_a"
        else:
            return Result("unparsed", note="second count not found")
        if len(ints) > len(used):
            note = (note + "; " if note else "") + "stated counts do not add up"
    A0 = Na * a1 + Nb * a2
    B0 = Na * b1 + Nb * b2
    # The shown coefficients are exact (the generator rounds them before use), so only the 2-dp rounding of
    # the options themselves needs tolerance: tight bounds keep near-miss distractors apart.
    rel, tol_a = 1e-4, 0.02
    if qtype == "numerical":
        others = [x for x in ms if x.start() not in used and x.start() > fpos[0]]
        dollar = [x for x in others if rest[max(0, x.start() - 2):x.start()].strip().endswith("$")]
        pick = dollar or others
        if not pick:
            return Result("unparsed", note="price not found")
        P = val(pick[0])
        neg = [i for i, (a, b) in enumerate(((a1, b1), (a2, b2))) if a + b * P < 0]
        if neg:
            note = (note + "; " if note else "") + f"individual function(s) {neg} negative at P={P}"
        return numeric_result(A0 + B0 * P, options, rel=rel, abs_=tol_a, note=note)
    if qtype == func_type:
        def same(o):
            p = _lin(o)
            if p is None:
                return None
            return close(p[0], A0, rel, tol_a) and close(p[1], B0, rel, tol_a)
        vals = [_lin(o) for o in options]
        if any(v is None for v in vals):
            return Result("unparsed", note="option not linear function")
        corr = [same(o) for o in options]
        dist = [abs(v[0] - A0) / max(1, abs(A0)) + abs(v[1] - B0) / max(1, abs(B0)) for v in vals]
        sgn = "+" if B0 >= 0 else "-"
        return Result("ok", f"{A0:.2f} {sgn} {abs(B0):.2f}P", corr, min(range(len(vals)), key=dist.__getitem__),
                      note)
    return Result("unparsed", note=f"unknown type {qtype}")


@register("aggregation_of_capital_demand")
def aggregation_of_capital_demand(q, options, qtype):
    """Market capital demand = horizontal sum of unit demands (see _aggregation)."""
    return _aggregation(q, options, qtype, "demand_function")


@register("aggregation_of_labor_supply")
def aggregation_of_labor_supply(q, options, qtype):
    """Market labour supply = horizontal sum of unit supplies (see _aggregation)."""
    return _aggregation(q, options, qtype, "supply_function")


@register("aggregation_of_offer_curve")
def aggregation_of_offer_curve(q, options, qtype):
    """Aggregate offer (supply) curve = horizontal sum of the regional curves (see _aggregation)."""
    return _aggregation(q, options, qtype, "offer_curve")

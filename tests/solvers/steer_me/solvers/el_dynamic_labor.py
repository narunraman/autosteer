"""Solvers for intertemporal / risk / labour-market / monopoly-DWL elements.

Elements: intertemporal_consumption_smoothing, state_based_consumption, price_of_risk,
labor_supply_distortions, labor_supply_elasticity, deadweight_loss.

Rounding: parameters in the question text are rounded (normally to 2 dp) while fillers compute the answer
from unrounded draws.  Instead of a flat relative tolerance, most solvers here evaluate the answer over the
box of parameter values consistent with the shown digits (a number printed with d>=1 decimals is taken as
exact to +-0.5*10^-max(d,2); integers are exact) and accept an option inside that interval, widened by
option rounding (+-0.0051) and 0.2% slack.  Where the interval is wide (e.g. a variance of 0.02 shown to
2 dp), several options may legitimately match: that is reported as multi_correct, i.e. the question is
ambiguous as printed.
"""
from __future__ import annotations

import itertools
import math
import re

from .common import NUM, Result, register

# ------------------------------------------------------------------------------------------ helpers
_TOK = re.compile(rf"(?<![\w.])({NUM})(\s*%)?")


def _hw(raw: str) -> float:
    """Half-width of the rounding interval implied by how a number is printed."""
    raw = raw.lower().split("e")[0]
    if "." not in raw:
        return 0.0
    d = len(raw.split(".")[1])
    return 0.5 * 10 ** (-max(d, 2))


class P:
    """A shown parameter: value, rounding half-width, and position in the text."""

    def __init__(self, raw: str, pct: bool = False, pos: int = -1):
        self.raw, self.pos = raw, pos
        v, h = float(raw), _hw(raw)
        self.v, self.h = (v / 100, h / 100) if pct else (v, h)

    def __repr__(self):
        return f"P({self.v}+-{self.h})"


def _tokens(q: str) -> list[P]:
    return [P(m.group(1), bool(m.group(2)), m.start(1)) for m in _TOK.finditer(q)]


def _after(q: str, kw: str, gap: int = 60, pct_ok: bool = True) -> P | None:
    """First number within `gap` non-digit characters after regex `kw`."""
    # the gap may not contain digits or cross a sentence end ('. ', '?', '!')
    m = re.search(rf"(?:{kw})(?:(?!\.\s)[^0-9?!]){{0,{gap}}}?(?<![\w.])({NUM})(\s*%)?", q, re.I | re.S)
    if not m:
        return None
    return P(m.group(1), pct_ok and bool(m.group(2)), m.start(1))


def _first(q: str, kws: list[str], gap: int = 60, pct_ok: bool = True, pred=None) -> P | None:
    """First number after any keyword pattern (patterns in priority order, then text order) passing `pred`."""
    for kw in kws:
        for m in re.finditer(rf"(?:{kw})", q, re.I | re.S):
            x = _after(q[m.start():], kw, gap, pct_ok)
            if x is not None and (pred is None or pred(x.v)):
                x.pos += m.start()
                return x
    return None


def _span(f, params: list[P]) -> tuple:
    """(center, lo, hi) of f over all corners of the rounding box (f may return a scalar or a tuple)."""
    center = f(*[p.v for p in params])
    vec = isinstance(center, (tuple, list))
    c = list(center) if vec else [center]
    lo, hi = list(c), list(c)
    free = [i for i, p in enumerate(params) if p.h > 0]
    for signs in itertools.product((-1, 1), repeat=len(free)):
        vals = [p.v for p in params]
        for i, s in zip(free, signs):
            vals[i] += s * params[i].h
        try:
            r = f(*vals)
        except (ZeroDivisionError, ValueError, OverflowError):
            continue
        r = list(r) if vec else [r]
        lo = [min(a, b) for a, b in zip(lo, r)]
        hi = [max(a, b) for a, b in zip(hi, r)]
    return c, lo, hi, vec


def _opt_vec(o: str) -> list[float]:
    """Numbers in an option, ignoring 'C1 =' labels and numpy reprs like 'np.float64(1.2)'."""
    o = re.sub(r"np\.\w+\(", "(", o)
    o = re.sub(r"\b[A-Za-z]+\d*\s*=", " ", o)
    return [float(x) for x in re.findall(rf"(?<![\w.]){NUM}", o.replace(",", " , "))]


def _inside(v, lo, hi, rel=0.002, abs_=0.0051) -> bool:
    s = max(abs_, rel * max(abs(lo), abs(hi)))
    return lo - s <= v <= hi + s


def interval_result(f, params: list[P], options: list[str], note: str = "", rel: float = 0.002) -> Result:
    """Mark options lying inside the answer interval implied by rounded parameters."""
    try:
        c, lo, hi, vec = _span(f, params)
    except (ZeroDivisionError, ValueError, OverflowError) as ex:
        return Result("undefined", note=f"{note} ({type(ex).__name__})")
    if not all(map(math.isfinite, c)):
        return Result("undefined", note=note or "non-finite answer")
    vecs = [_opt_vec(o) for o in options]
    if any(len(v) != len(c) for v in vecs):
        return Result("unparsed", note=note or "option arity mismatch")
    corr = [all(_inside(x, a, b, rel) for x, a, b in zip(v, lo, hi)) for v in vecs]
    dist = [sum(abs(x - y) / max(1.0, abs(y)) for x, y in zip(v, c)) for v in vecs]
    closest = min(range(len(vecs)), key=dist.__getitem__)
    exp = tuple(round(x, 6) for x in c) if vec else round(c[0], 6)
    rng = f" range=[{', '.join(f'{a:.4g}..{b:.4g}' for a, b in zip(lo, hi))}]"
    return Result("ok", exp, corr, closest, (note + rng).strip())


# ------------------------------------------------------------------ intertemporal_consumption_smoothing
def _euler_path(W, incomes, beta, r, gamma):
    """CRRA u(c)=c^(1-g)/(1-g), max sum beta^t u(c_t) s.t. sum c_t/(1+r)^t = W + sum y_t/(1+r)^t.
    Euler: c_{t+1}/c_t = (beta(1+r))^(1/gamma)."""
    g = (beta * (1 + r)) ** (1 / gamma)
    pv = W + sum(y / (1 + r) ** t for t, y in enumerate(incomes))
    c1 = pv / sum((g / (1 + r)) ** t for t in range(len(incomes)))
    return tuple(c1 * g ** t for t in range(len(incomes)))


@register("intertemporal_consumption_smoothing")
def intertemporal_consumption_smoothing(q, options, qtype):
    """Two(+)-period CRRA consumption smoothing.

    Budget: wealth W and income y_t available from period 1, savings earn r between periods, so
    PV(c) = W + y_1 + y_2/(1+r) (+ ...).  Euler: C2/C1 = (beta(1+r))^(1/gamma).
    ASSUMPTION: the text calls the 0.9-0.99 parameter a "discount *rate*" rho, so beta = 1/(1+rho) (literal
    reading).  If the text says "discount factor", beta = value.  The note reports which options would match
    under the other reading (beta = shown value) and under a flat path c_1 = c_2 (the numerical filler's
    output in practice).  Both question types ('closed_form', 'numerical') have the same economics; only the
    filler differs.  Returns "undefined" when the text has no risk-aversion coefficient (some templates label
    gamma as a second "social discount rate" / "preference factor"), since the CRRA curvature is then unknown.
    Tolerance is the rounding interval + 0.2% (not the default 2%), because the flat-path answers are often
    within 1-2% of the Euler solution.
    """
    r = _first(q, [r"interest rate|return rate|rate of return|\binterest\b", r"growth rate|investment rate",
                   r"\bearn\w*"], 70, pred=lambda v: 0 <= v < 1)
    g = _after(q, r"risk[- ]aversion|aversion to risk|\bgamma\b|risk preference", 70)  # me-fillers: + "aversion to risk"
    dms = list(re.finditer(r"discount(?:s|ing)?\b(?:(?!\.\s)[^?0-9]){0,40}?\b(rate|factor)", q, re.I))
    if not (r and dms):
        return Result("unparsed", note="missing interest rate / discount")
    if not g:
        # e.g. 'a discount rate of 0.96 ... factoring in a social discount rate of 2.96': the CRRA coefficient
        # is mislabelled (a second discount rate / 'preference factor'), so the problem is not identified.
        return Result("undefined", note="no risk-aversion coefficient in text (gamma mislabelled, e.g. "
                                         f"{len(dms)} 'discount rate/factor' mentions)")
    dm = dms[0]
    d = _after(q[dm.start():], r"discount(?:s|ing)?\b(?:(?!\.\s)[^?0-9]){0,40}?\b(?:rate|factor)", 70)
    if d is None:
        return Result("unparsed", note="discount value not found")
    d.pos += dm.start()
    money = [t for t in _tokens(q) if t.v >= 100]
    W = _after(q, r"wealth|endowment|initial|starting|start(?:s|ing)? with|reserve|fund of", 40)
    if W is None or W.v < 100:
        W = money[0] if money else None
    if W is None:
        return Result("unparsed", note="wealth not found")
    used = {W.pos, r.pos, g.pos, d.pos}
    # incomes: money-sized numbers (>= 100) other than W, in text order
    inc = [t for t in money if t.pos not in used]
    if len(inc) < 2:
        return Result("unparsed", note="incomes not found")
    n_opt = len(_opt_vec(options[0]))
    if n_opt != len(inc):
        return Result("unparsed", note=f"{len(inc)} incomes vs {n_opt}-component options")
    factor_word = dm.group(1).lower() == "factor"

    def f_lit(Wv, rv, gv, dv, *ys):
        beta = dv if factor_word else 1 / (1 + dv)
        return _euler_path(Wv, ys, beta, rv, gv)

    def f_alt(Wv, rv, gv, dv, *ys):
        beta = 1 / (1 + dv) if factor_word else dv
        return _euler_path(Wv, ys, beta, rv, gv)

    params = [W, r, g, d, *inc]
    alt = interval_result(f_alt, params, options)
    eq = interval_result(lambda Wv, rv, gv, dv, *ys: _equal_path(Wv, ys, rv), params, options)
    conv = "beta=value" if factor_word else "beta=1/(1+rate)"
    note = (f"{conv}; alt({'beta=1/(1+v)' if factor_word else 'beta=value'}) matches "
            f"{[i for i, c in enumerate(alt.correct or []) if c]}; equal-consumption path matches "
            f"{[i for i, c in enumerate(eq.correct or []) if c]}")
    return interval_result(f_lit, params, options, note=note)


def _equal_path(W, ys, r):
    """Flat path c_t = c with the same PV (what beta(1+r)=1 would give) - diagnostic only."""
    pv = W + sum(y / (1 + r) ** t for t, y in enumerate(ys))
    c = pv / sum(1 / (1 + r) ** t for t in range(len(ys)))
    return (c,) * len(ys)


# ------------------------------------------------------------------------------ state_based_consumption
@register("state_based_consumption")
def state_based_consumption(q, options, qtype):
    """Ex-ante allocation of wealth W over goods x, y with state-dependent prices (states A, B, probs pA, pB).

    Textbook reading used (the same model the template describes): the consumer commits dollar spending
    (s_x, s_y), s_x + s_y = W, before the state is known; in state i she gets s_x/p_ix and s_y/p_iy units.
      * Cobb-Douglas x^a y^b: E[u] = a ln s_x + b ln s_y + const, so s_x = a/(a+b) W, s_y = b/(a+b) W
        (independent of prices and probabilities).
      * Linear a x + b y: E[u] = s_x a E[1/p_x] + s_y b E[1/p_y]; everything goes to the good with the larger
        a E[1/p_x] vs b E[1/p_y] (a corner).
    The answer is a pair of dollar amounts. (The old filler printed 2-character fragments such as '(5, 8)'; it was
    fixed in the audit, and the note no longer decodes them.) Prices are read state by state, or good by good when
    a state label sits between the first two prices.
    """
    ucd = re.search(rf"x\^\(?({NUM})\)?\s*\*?\s*y\^\(?({NUM})\)?", q)
    ulin = re.search(rf"U\(x,\s*y\)\s*=\s*({NUM})\s*\*?\s*x\s*\+\s*({NUM})\s*\*?\s*y", q)
    if not (ucd or ulin):
        return Result("unparsed", note="utility not found")
    um = ucd or ulin
    a, b = float(um.group(1)), float(um.group(2))
    rest = q[:um.start()] + " " * (um.end() - um.start()) + q[um.end():]
    # me-fillers: labels such as 'Scenario 1', 'State 2', '3D' are not parameters
    rest = re.sub(r"\b(?:scenario|state|outcome|case|option)\s*\d\b|\b\d+D\b|\b[A-Za-z]+-\d+\b",
                  lambda z: " " * len(z.group(0)), rest, flags=re.I)
    toks = _tokens(rest)
    if not toks:
        return Result("unparsed", note="no numbers")
    W = max(toks, key=lambda t: t.v)          # wealth is the only large number (~1e5)
    toks = [t for t in toks if t is not W]
    probs = [t for t in toks if 0 <= t.v <= 1]
    pr = None
    for i in range(len(probs) - 1):
        if abs(probs[i].v + probs[i + 1].v - 1) < 0.02:
            pr = (probs[i].v, probs[i + 1].v)
            used = {probs[i].pos, probs[i + 1].pos}
            break
    if pr is None:
        return Result("unparsed", note="probabilities not found")
    prices = [t for t in toks if t.pos not in used]
    if len(prices) < 4 or W.v < 1000:
        return Result("unparsed", note="wealth/prices not found")
    pAx, pAy, pBx, pBy = (p.v for p in prices[:4])
    # Some templates list the prices good by good ("x costs 3.9 in Scenario 1 and 6.33 in Scenario 2, while y costs
    # 9.32 in Scenario 1 and ..."), not state by state: then a state label sits between the first two prices.
    if re.search(r"\b(?:state|scenario|situation|case|outcome|condition)\s*(?:[AB12]|one|two)\b|"
                 r"\b(?:first|second) (?:state|scenario|situation|case|outcome|condition)\b",
                 q[prices[0].pos:prices[1].pos], re.I):
        pAx, pBx, pAy, pBy = pAx, pAy, pBx, pBy
    if ucd:
        exp = (a / (a + b) * W.v, b / (a + b) * W.v)
    else:
        mx = a * (pr[0] / pAx + pr[1] / pBx)
        my = b * (pr[0] / pAy + pr[1] / pBy)
        exp = (W.v, 0.0) if mx > my else (0.0, W.v)
    corr = []
    for o in options:
        v = _opt_vec(o)
        corr.append(len(v) == 2 and all(abs(x - y) <= max(0.0051, 0.002 * abs(y)) for x, y in zip(v, exp)))
    note = f"prices xA={pAx} yA={pAy} xB={pBx} yB={pBy}"
    return Result("ok", tuple(round(x, 2) for x in exp), corr, None, note)


# ---------------------------------------------------------------------------------------- price_of_risk
# Code templates (review, October 2026): U = E[r] - (A/2) Var(r); w* = (mu - rf) / (A sigma^2) = S / (A sigma);
# clipped to [0, 100%] when the question says borrowing is not allowed. Answers in percent / percentage points.
def _num_after(q: str, pattern: str) -> P | None:
    m = re.search(rf"{pattern}\s*\$?({NUM})", q, re.S)
    return P(m.group(1), False, m.start(1)) if m else None


@register("price_of_risk")
def price_of_risk(q, options, qtype):
    A = _num_after(q, r"risk aversion A =")
    if A is None:
        return Result("unparsed", note="no A")
    clip = "cannot borrow" in q
    if "expected value of" in q:
        rf, mu, sd = (_num_after(q, k) for k in (r"paying", r"expected value of", r"standard deviation of"))
        if None in (rf, mu, sd):
            return Result("unparsed", note="allocation parameters")

        def f(a, r_, m_, s_):
            w = (m_ - r_) / 100 / (a * (s_ / 100) ** 2)
            return 100 * (min(1.0, max(0.0, w)) if clip else w)
        return interval_result(f, [A, rf, mu, sd], options, "w*=(mu-rf)/(A s^2)" + (" clipped" if clip else ""))
    sd, s0, s1 = (_num_after(q, k) for k in (r"standard deviation is", r"rises from", r"rises from [\d.]+ to"))
    if None in (sd, s0, s1):
        return Result("unparsed", note="analysis parameters")
    if "percentage points" in q:
        return interval_result(lambda a, s, x0, x1: 100 * (x1 - x0) / (a * s / 100), [A, sd, s0, s1], options,
                               "dw=(S1-S0)/(A s)")
    return interval_result(lambda a, s, x0, x1: 100 * x1 / (a * s / 100), [A, sd, s0, s1], options, "w1=S1/(A s)")


# ----------------------------------------------------------------------------- labor_supply_distortions
# Code templates (review, October 2026).
#   hours_worked: %dH = -e t (linear approximation, gross wage fixed, tax from zero), in percent.
#   labor_supply: LS = a + b(W - T), W fixed, T0 -> T1: dLS = -b (T1 - T0).
#   equilibrium_wage_quantity: a + b(W - T) = c - dW: W = (c - a + bT) / (b + d), Q = c - dW.
@register("labor_supply_distortions")
def labor_supply_distortions(q, options, qtype):
    if "income tax of" in q:
        t, e = _num_after(q, r"income tax of"), _num_after(q, r"after-tax wage, is")
        if None in (t, e):
            return Result("unparsed", note="hours parameters")
        return interval_result(lambda tv, ev: -ev * tv, [t, e], options, "%dH=-e t")
    m = re.search(rf"LS = ({NUM}) \+ ({NUM})\(W - T\)", q)
    if not m:
        return Result("unparsed", note="no LS")
    a, b = P(m.group(1)), P(m.group(2))
    if "rises from" in q:
        t0, t1 = _num_after(q, r"rises from"), _num_after(q, r"rises from \$[\d.]+ to")
        if None in (t0, t1):
            return Result("unparsed", note="tax change")
        return interval_result(lambda bv, x0, x1: -bv * (x1 - x0), [b, t0, t1], options, "dLS=-b dT")
    m = re.search(rf"LD = ({NUM}) - ({NUM})W", q)
    T = _num_after(q, r"The tax is")
    if not m or T is None:
        return Result("unparsed", note="no LD / tax")
    c, d = P(m.group(1)), P(m.group(2))

    def f(av, bv, cv, dv, tv):
        W = (cv - av + bv * tv) / (bv + dv)
        return (W, cv - dv * W)
    return interval_result(f, [a, b, c, d, T], options, "tax wedge equilibrium")


# ------------------------------------------------------------------------------ labor_supply_elasticity
# Code templates (review, October 2026): %dL = e %dW (linear approximation), answers in percent.
@register("labor_supply_elasticity")
def labor_supply_elasticity(q, options, qtype):
    e = _num_after(q, r"labor supply of [^:]*? is")
    m = re.search(rf"wage (rises|falls) by ({NUM})%", q)
    if e is not None and m:
        x = P(m.group(2))
        s = 1 if m.group(1) == "rises" else -1
        return interval_result(lambda ev, xv: s * ev * xv, [e, x], options, "%dL=e %dW")
    m = re.search(rf"to (rise|fall) by ({NUM})%", q)
    if e is not None and m:
        y = P(m.group(2))
        s = 1 if m.group(1) == "rise" else -1
        return interval_result(lambda ev, yv: s * yv / ev, [e, y], options, "%dW=%dL/e")
    return Result("unparsed", note="no elasticity / change")


# -------------------------------------------------------------------------------------- deadweight_loss
@register("deadweight_loss")
def deadweight_loss(q, options, qtype):
    """Monopoly DWL with linear inverse demand P = a - bQ and supply / marginal cost P = c + dQ.

    Competitive: Q* = (a - c)/(b + d).  Monopoly (compute_monopoly_price): MR = a - 2bQ = MC ->
    Qm = (a - c)/(2b + d), Pm = a - b Qm.  If a monopoly price Pm is given (monopoly_price_given):
    Qm = (a - Pm)/b.  DWL = 1/2 (Pm - MC(Qm)) (Q* - Qm).
    """
    curves = list(re.finditer(rf"({NUM})\s*\*?\s*Q\s*\+\s*({NUM})", q))
    if len(curves) < 2:
        return Result("unparsed", note="two linear curves not found")
    (d0, d1), (s0, s1) = [(P(m.group(1)), P(m.group(2))) for m in curves[:2]]
    if d0.v > 0 and s0.v < 0:
        (d0, d1), (s0, s1) = (s0, s1), (d0, d1)
    if d0.v == 0:
        return Result("undefined", note="demand slope prints as 0 (rounded): DWL not identified from the text")
    if d0.v > 0 or s0.v < 0:
        return Result("unparsed", note="slopes have unexpected signs")
    # demand is printed as '-bQ + a'; we work with b = -slope
    rest = q
    for m in reversed(curves[:2]):
        rest = rest[:m.start()] + " " * (m.end() - m.start()) + rest[m.end():]
    left = _tokens(rest)
    pm = left[0] if len(left) == 1 else _after(rest, r"monopoly price|price", 40)
    given = qtype == "monopoly_price_given" or (not qtype and pm is not None)

    def qstar(ds, di, ss, si):
        return (di - si) / (ss - ds)

    if given:
        if pm is None:
            return Result("unparsed", note="monopoly price not found")

        def f(ds, di, ss, si, p):
            Qs, Qm = qstar(ds, di, ss, si), (di - p) / (-ds)
            return 0.5 * (p - (ss * Qm + si)) * (Qs - Qm)
        return interval_result(f, [d0, d1, s0, s1, pm], options, note="Pm given")

    def g(ds, di, ss, si):
        Qs, Qm = qstar(ds, di, ss, si), (di - si) / (ss - 2 * ds)
        P_m = di + ds * Qm
        return 0.5 * (P_m - (ss * Qm + si)) * (Qs - Qm)
    return interval_result(g, [d0, d1, s0, s1], options, note="MR=MC")

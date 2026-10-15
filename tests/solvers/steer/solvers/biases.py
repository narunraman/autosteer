"""Phase 2 solvers / consistency hooks for the behavioural-bias group:
certainty_effect, ambiguity_aversion, endowment_effect (all consistency-graded, correct_index = null).

Consistency encoding (see the element yamls for the full rule text):
  consistency_group  rows that are scored jointly
  consistency_rule   "<rule_id>:<allowed>" where <allowed> lists the consistent choice(s) as option indices
                     into the *released* (final, already shuffled) `options` of the rows of the group,
                     ordered by part.  Examples
                       "eu_implied:1"          one-row group, option 1 is the only EU-consistent choice
                       "ellsberg_seu:01,10"    two-row group (part 0, part 1); consistent iff the pair of
                                               chosen indices is (0,1) or (1,0)
                       "flip_endowment:0"      endowment row; option 0 is the choice consistent with the
                                               stated previous decision (group = same scenario)
  The options of these elements are shuffled here (seeded by question_id) *before* the rule is written, and
  POLICY.shuffle is False so that repair.py does not reshuffle afterwards.
"""
from __future__ import annotations

import hashlib
import re
from fractions import Fraction

import numpy as np
import pandas as pd

REPAIRS = {
    "consistency_shuffle": "options shuffled (seeded by question_id) before the consistency rule indices were "
                           "written (biases.py); POLICY.shuffle=False so repair.py does not reshuffle",
    "stack_split": "stacked copies of the element separated (endowment_effect accept/reject); type is the "
                   "condition, group = scenario",
}


def _out(index, pred, status) -> pd.DataFrame:
    return pd.DataFrame({"pred": pd.array(pred, dtype="Int64"), "status": list(status)}, index=index)


def _seed(qid: str) -> int:
    return int.from_bytes(hashlib.sha256(qid.encode()).digest()[:8], "little")


def _perm(qid: str, n: int) -> np.ndarray:
    return np.random.default_rng(_seed(qid)).permutation(n)


def _shuffle_rows(df: pd.DataFrame, el: str) -> tuple[pd.DataFrame, dict]:
    """Shuffle options of every row (seed = final question_id); return df and {index: perm}."""
    perms = {}
    new = []
    for i, b, p, o in zip(df.index, df.base_id, df.part, df.options):
        perm = _perm(f"{el}/{b}/{p}", len(o))
        perms[i] = perm
        new.append([o[k] for k in perm])
    df = df.copy()
    df["options"] = new
    df["repair"] = [r + ["consistency_shuffle"] for r in df.repair]
    return df, perms


def _new_index(perm: np.ndarray, old: int) -> int:
    return int(np.flatnonzero(perm == old)[0])


# =============================================================================================== certainty
_TOK = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%)?")


def _tokens(s: str):
    return [(Fraction(m.group(1)), m.group(2) is not None, m.start()) for m in _TOK.finditer(s)]


def _split_probs(toks):
    """Find the complementary probability pair (two % tokens summing to 100). Returns (prob_toks, value_toks)."""
    pct = [i for i, t in enumerate(toks) if t[1]]
    pairs = [(i, j) for a, i in enumerate(pct) for j in pct[a + 1:] if abs(toks[i][0] + toks[j][0] - 100) < Fraction(2, 100)]
    if len(pairs) != 1:
        return None, None
    i, j = pairs[0]
    probs = [toks[i], toks[j]]
    vals = [t for k, t in enumerate(toks) if k not in (i, j)]
    return probs, vals


def _prob_of(val_tok, probs):
    """Probability attached to a value: the first probability token after it, else the last one before it."""
    after = [p for p in probs if p[2] > val_tok[2]]
    if after:
        return min(after, key=lambda p: p[2])[0] / 100
    return max(probs, key=lambda p: p[2])[0] / 100


def parse_lottery_option(s: str):
    """Option 'v with probability q and 0 with probability 1-q' -> (v, q) or None."""
    toks = _tokens(s)
    probs, vals = _split_probs(toks)
    if probs is None:
        return None
    nz = [t for t in vals if t[0] != 0]
    if len(nz) != 1:
        return None
    return nz[0][0], _prob_of(nz[0], probs)


def parse_certainty(q: str, options: list[str]):
    """Returns dict(X, Y, p, opts=[(v,q),(v,q)], c=index of the option paying X) or a status string."""
    if len(options) != 2:
        return "unparsed"
    lots = [parse_lottery_option(o) for o in options]
    if any(l is None for l in lots):
        return "ill_posed:option_unparsed"
    toks = _tokens(q)
    probs, vals = _split_probs(toks)
    if probs is None:
        return "ill_posed:prior_gamble_missing"
    nz = [t for t in vals if t[0] != 0]
    if len(nz) != 2:
        return "ill_posed:prior_gamble_missing"
    qvals = [t[0] for t in nz]
    hits = [k for k, (v, _) in enumerate(lots) if v in qvals]
    if len(hits) == 2:  # both option values occur in the question: C is the one equal to a value without a prob
        hits = [k for k in hits if lots[k][0] != lots[1 - k][0]] or hits
    if len(hits) != 1:
        return "ill_posed:certain_option_ambiguous"
    c = hits[0]
    X = lots[c][0]
    rest = [t for t in nz if t[0] != X] or [nz[1]]  # X == Y possible
    ytok = rest[0]
    Y, p = ytok[0], _prob_of(ytok, probs)
    return {"X": X, "Y": Y, "p": p, "opts": lots, "c": c}


def eu_implied(X, Y, p, C, D):
    """Given the stated strict preference  X (sure) > (Y, p; 0, 1-p),  u increasing with u(0)=0, which of the
    lotteries C=(xc, qc; 0) and D=(xd, qd; 0) is preferred by *every* such expected-utility maximiser?
    Returns "C", "D" or None (not implied)."""
    (xc, qc), (xd, qd) = C, D

    def forced(a, b):  # is lottery a strictly preferred to b for all admissible u?
        (xa, qa), (xb, qb) = a, b
        if qb == 0 or xb == 0:
            return qa > 0 and xa > 0
        if qa == 0 or xa == 0:
            return False
        if xb <= xa:  # u(xb) <= u(xa)
            return qa > qb or (qa == qb and xb < xa)
        # xb > xa: bound on u(xb)/u(xa) only from the stated preference, if a pays X and xb <= Y
        if xa == X and xb <= Y and p > 0:
            return qb <= p * qa  # sup u(xb)/u(xa) = 1/p (not attained)
        return False

    if forced(C, D):
        return "C"
    if forced(D, C):
        return "D"
    return None


def solve_certainty(df: pd.DataFrame) -> pd.DataFrame:
    pred, status = [], []
    for q, o in zip(df.question_text, df.options):
        o = list(o)
        r = parse_certainty(q, o)
        if isinstance(r, str):
            pred.append(None)
            status.append(r)
            continue
        c = r["c"]
        imp = eu_implied(r["X"], r["Y"], r["p"], r["opts"][c], r["opts"][1 - c])
        if imp is None:
            pred.append(None)
            status.append("ill_posed:not_implied_by_stated_choice")
        else:
            pred.append(c if imp == "C" else 1 - c)
            status.append("ok")
    return _out(df.index, pred, status)


def consistency_certainty(df: pd.DataFrame) -> pd.DataFrame:
    keep = df["drop"].isna()
    r = solve_certainty(df[["base_id", "part", "question_text", "options", "type", "domain", "tags", "difficulty"]])
    df, perms = _shuffle_rows(df, "certainty_effect")
    rule = []
    for i, pr, st in zip(df.index, r.pred, r.status):
        if st == "ok" and pd.notna(pr):
            rule.append(f"eu_implied:{_new_index(perms[i], int(pr))}")
        else:
            rule.append(None)
    df["consistency_group"] = "certainty_effect/" + df.base_id
    df["consistency_rule"] = rule
    bad = keep & df.consistency_rule.isna()
    df.loc[bad, "drop"] = "no_consistency_rule"
    return df


# =============================================================================================== ambiguity
_STOP = set("""a an the and or of to if you your yours is are be it its it's in on at for from with by as
that this these those which who whom will would can could may might get gets got receive receives received win
wins earn gain obtain collect accept given give nothing zero no none not otherwise either any other else
but also only then than there their they them he she his her we our us i me my one two
free complimentary charge cost price regular normal full pay paid prize reward rewarded
select selecting choose chooses choosing chose pick picks picked picking draw draws drawn drawing pull
pulling out opt decide order find finds catch encounter selection selected turns turn turned ends up being
end correctly incorrectly wrong incorrect correct do don't does fail so type kind color colour choice
""".split())


def _stem(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and w[:-2].endswith(("ss", "sh", "ch", "x")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _word_pos(s: str) -> dict[str, int]:
    """content word -> first character position"""
    out = {}
    for m in re.finditer(r"[a-z]+(?:'[a-z]+)?", s.lower()):
        w = m.group(0)
        if w in _STOP or len(w) < 3:
            continue
        out.setdefault(_stem(w), m.start())
    return out


def _words(s: str) -> set[str]:
    return set(_word_pos(s))


_NUM = re.compile(r"\d+(?:\.\d+)?")
_BOXNUM = re.compile(r"\b(?:Box|Chest|Bag|Urn|Jar|box)\s+\d\b|^\s*[A-D]\.\s")


def _nums(o: str) -> list[float]:
    return [float(x) for x in _NUM.findall(_BOXNUM.sub(" ", o))]


def _stakes(o: str) -> tuple:
    return tuple(sorted(x for x in _nums(o) if x != 0))


_PRIZE = re.compile(r"\b(?:free|complimentary)\s+((?:[a-z]+\s+){0,3}?[a-z]+)\s+(?:if|whether|only|when|for|,)"
                    r"|\b((?:[a-z]+\s+){0,2}?[a-z]+)\s+(?:for free|free of charge|at no cost)\b")


def _prize(o: str):
    m = _PRIZE.search(o.lower())
    if not m:
        return None
    ws = [_stem(x) for x in (m.group(1) or m.group(2)).split() if x not in ("a", "an", "the")]
    if m.group(2) is not None:
        ws = ws[-1:]  # "<noun> for free": only the head noun
    if not ws or ws[-1] in ("it", "one", "them", "treat", "item"):
        return None
    return " ".join(ws)


def _counts(t0: str):
    """(k, U) = number of known-colour items and of ambiguous items, from the part-0 text."""
    n = [int(x) for x in re.findall(r"\d+", _BOXNUM.sub(" ", t0))]
    if len(n) == 2 and n[0] > 0 and n[1] > 0:
        return n[0], n[1]
    if len(n) == 3 and n[0] == n[1] + n[2] and n[1] > 0:
        return n[1], n[2]
    return None


def _rationalizable(ev0, ev1, k, U):
    """ev0/ev1: part-0/part-1 options as frozensets over {'K','U1','U2'}. Beliefs: P(K)=k/N, P(U1)=x,
    P(U2)=U/N-x, x in (0, U/N). Returns the set of (i, j) choice pairs that are *strict* SEU best responses
    for a common x."""
    N = Fraction(k + U)
    pk, pu = Fraction(k) / N, Fraction(U) / N

    def P(ev, x):
        return (pk if "K" in ev else 0) + (x if "U1" in ev else 0) + ((pu - x) if "U2" in ev else 0)

    def diff(evs, x):
        return P(evs[0], x) - P(evs[1], x)

    # critical points: roots of the two linear differences
    crit = {Fraction(0), pu}
    for evs in (ev0, ev1):
        a, b = diff(evs, Fraction(0)), diff(evs, pu)
        if a != b:
            r = a / (a - b) * pu
            if 0 < r < pu:
                crit.add(r)
    pts = sorted(crit)
    ok = set()
    for lo, hi in zip(pts, pts[1:]):
        x = (lo + hi) / 2
        d0, d1 = diff(ev0, x), diff(ev1, x)
        if d0 == 0 or d1 == 0:
            continue
        ok.add((0 if d0 > 0 else 1, 0 if d1 > 0 else 1))
    return ok


def ellsberg_structure(t0: str, a: list[str], b: list[str]):
    """Return (pairs, status). pairs = allowed (i, j) index pairs in the *given* option order.

    Three colours: K (known count k), U1/U2 (ambiguous, U items in unknown proportion).  The part-1 options
    each bet on two colours; the three colour word-sets are C = words shared by both part-1 options and
    D0/D1 = words distinguishing them.  Each part-0 option must map to exactly one of the three; K is the colour
    mentioned first in the part-0 text.  A pair of choices is consistent iff some single belief x = P(U1)
    in (0, U/N) makes both choices strict subjective-expected-utility best responses (same prize/stakes in
    both options of a part).  For the standard sure-thing construction this gives {(K, K u U2), (U1, U1 u U2)}.
    """
    if len(a) != 2 or len(b) != 2:
        return None, "unparsed"
    for opts in (a, b):
        if _stakes(opts[0]) != _stakes(opts[1]):
            return None, "ill_posed:unequal_stakes"
        nums = _nums(opts[0])
        if (len(nums) >= 2 and nums[0] == nums[1]) or re.search(r"\b(?:a|an)\s+\w+\s.*\b1\s+(?:\w+\s+)?otherwise", opts[0]):
            return None, "ill_posed:win_equals_lose"
        p0, p1 = _prize(opts[0]), _prize(opts[1])
        if p0 and p1 and p0 != p1:
            return None, "ill_posed:unequal_prizes"
    cnt = _counts(t0)
    if cnt is None:
        return None, "ill_posed:counts_missing"
    pos = _word_pos(t0)
    vocab = set(pos)
    wa = [_words(x) & vocab for x in a]
    wb = [_words(x) & vocab for x in b]
    cats = {"C": wb[0] & wb[1], "D0": wb[0] - wb[1], "D1": wb[1] - wb[0]}
    if not all(cats.values()):
        return None, "ill_posed:structure_unparsed"
    e = [wa[0] - wa[1], wa[1] - wa[0]]
    m = []
    for ei in e:
        # a colour bet on in part 0 appears either in exactly one part-1 option (D0/D1) or, for a
        # non-standard construction, in both (C); a D hit is more specific and wins over a C hit
        hit = [c for c in ("D0", "D1") if ei & cats[c]] or [c for c in ("C",) if ei & cats[c]]
        if len(hit) != 1:
            return None, "ill_posed:structure_unparsed"
        m.append(hit[0])
    if m[0] == m[1]:
        return None, "ill_posed:structure_unparsed"
    first = {c: min(pos[w] for w in ws) for c, ws in cats.items()}
    known = min(first, key=first.get)
    unk = sorted((c for c in cats if c != known), key=first.get)
    name = {known: "K", unk[0]: "U1", unk[1]: "U2"}
    ev1 = (frozenset({name["C"], name["D0"]}), frozenset({name["C"], name["D1"]}))
    ev0 = (frozenset({name[m[0]]}), frozenset({name[m[1]]}))
    if ev0[0] == ev0[1] or ev1[0] == ev1[1]:
        return None, "ill_posed:structure_unparsed"
    ok = _rationalizable(ev0, ev1, *cnt)
    if len(ok) == 4:
        return None, "ill_posed:no_restriction"
    if not ok:
        return None, "ill_posed:no_consistent_pair"
    return ok, "ok"


def _ambiguity_bases(df: pd.DataFrame):
    p0 = df[df.part == 0].set_index("base_id")
    p1 = df[df.part == 1].set_index("base_id")
    res = {}
    for b in p0.index:
        if b not in p1.index:
            res[b] = (None, "ill_posed:missing_part")
            continue
        res[b] = ellsberg_structure(p0.at[b, "question_text"], list(p0.at[b, "options"]), list(p1.at[b, "options"]))
    return res


def solve_ambiguity(df: pd.DataFrame) -> pd.DataFrame:
    """No single correct option: pred is always NA; status says whether the base has a well-defined
    SEU consistency rule ("ok") or is ill-posed."""
    res = _ambiguity_bases(df)
    status = [res.get(b, (None, "ill_posed:missing_part"))[1] for b in df.base_id]
    return _out(df.index, [None] * len(df), status)


def consistency_ambiguity(df: pd.DataFrame) -> pd.DataFrame:
    res = _ambiguity_bases(df)
    df, perms = _shuffle_rows(df, "ambiguity_aversion")
    idx = {(b, p): i for i, b, p in zip(df.index, df.base_id, df.part)}
    rule = pd.Series(None, index=df.index, dtype=object)
    for b, (pairs, st) in res.items():
        if st != "ok" or (b, 0) not in idx or (b, 1) not in idx:
            continue
        i0, i1 = idx[(b, 0)], idx[(b, 1)]
        new = sorted((_new_index(perms[i0], i), _new_index(perms[i1], j)) for i, j in pairs)
        r = "ellsberg_seu:" + ",".join(f"{i}{j}" for i, j in new)
        rule[i0] = r
        rule[i1] = r
    df["consistency_group"] = "ambiguity_aversion/" + df.base_id
    df["consistency_rule"] = rule
    bad = df["drop"].isna() & df.consistency_rule.isna()
    df.loc[bad, "drop"] = "no_consistency_rule"
    return df


# =============================================================================================== endowment
_ENDOW_PREFIX = [
    (re.compile(r"^\s*You previously answered no in the following situation:"), "no"),
    (re.compile(r"^\s*When in the following situation, you declined to trade\."), "no"),
    (re.compile(r"^\s*You previously answered yes in the following situation:"), "yes"),
    (re.compile(r"^\s*When you were in the following situation, you accepted the trade\."), "yes"),
]
_ENDOW_SPLIT = re.compile(r"'(?P<prev>.*)'\s*\n\s*You are now (?:given a trade request |given a proposition |)in a "
                          r"similar situation:\s*(?P<now>.*)\Z", re.S)


def parse_endowment(q: str):
    """-> (previous answer 'yes'/'no', previous situation, current situation) or None."""
    prev = None
    for rx, ans in _ENDOW_PREFIX:
        if rx.search(q):
            prev = ans
            break
    m = _ENDOW_SPLIT.search(q)
    if prev is None or not m:
        return None
    return prev, " ".join(m.group("prev").split()), " ".join(m.group("now").split())


def solve_endowment(df: pd.DataFrame) -> pd.DataFrame:
    """The current offer is the mirror of the previous one (you now own the item you were offered, and are
    offered the item you owned).  With no endowment effect, a strict preference revealed by the previous
    answer implies the opposite answer now: previously 'no' (kept A over B) -> now 'Yes' (take A for B);
    previously 'yes' -> now 'No'."""
    pred, status = [], []
    for q, o in zip(df.question_text, df.options):
        o = [x.strip().lower() for x in o]
        r = parse_endowment(q)
        if r is None:
            pred.append(None)
            status.append("unparsed")
            continue
        want = "yes" if r[0] == "no" else "no"
        hits = [i for i, x in enumerate(o) if x == want]
        pred.append(hits[0] if len(hits) == 1 else None)
        status.append("ok" if len(hits) == 1 else ("no_match" if not hits else "multi_match"))
    return _out(df.index, pred, status)


def endowment_scenario(q: str) -> str | None:
    r = parse_endowment(q)
    if r is None:
        return None
    return hashlib.sha1((r[1] + "\x1f" + r[2]).encode()).hexdigest()[:10]


def consistency_endowment(df: pd.DataFrame) -> pd.DataFrame:
    r = solve_endowment(df[["base_id", "part", "question_text", "options", "type", "domain", "tags", "difficulty"]])
    df, perms = _shuffle_rows(df, "endowment_effect")
    rule = []
    for i, pr, st in zip(df.index, r.pred, r.status):
        rule.append(f"flip_endowment:{_new_index(perms[i], int(pr))}" if st == "ok" else None)
    df["consistency_rule"] = rule
    df["consistency_group"] = ["endowment_effect/" + s if s else None for s in df.question_text.map(endowment_scenario)]
    bad = df["drop"].isna() & (df.consistency_rule.isna() | df.consistency_group.isna())
    df.loc[bad, "drop"] = "no_consistency_rule"
    return df


def fix_endowment(df: pd.DataFrame) -> pd.DataFrame:
    """Record the accept/reject stack split (type already carries it) on every row."""
    df = df.copy()
    df["repair"] = [r + ["stack_split"] for r in df.repair]
    return df


# =============================================================================================== registry
SOLVERS = {
    "certainty_effect": solve_certainty,
    "ambiguity_aversion": solve_ambiguity,
    "endowment_effect": solve_endowment,
}
POLICY = {
    "certainty_effect": {"grading": "consistency", "authoritative": False, "drop_status": ["ill_posed", "unparsed"],
                         "shuffle": False},
    "ambiguity_aversion": {"grading": "consistency", "authoritative": False,
                           "drop_status": ["ill_posed", "unparsed"], "shuffle": False},
    "endowment_effect": {"grading": "consistency", "authoritative": False, "drop_status": ["ill_posed", "unparsed"],
                         "shuffle": False},
}
FIXES = {"endowment_effect": fix_endowment}
CONSISTENCY = {
    "certainty_effect": consistency_certainty,
    "ambiguity_aversion": consistency_ambiguity,
    "endowment_effect": consistency_endowment,
}

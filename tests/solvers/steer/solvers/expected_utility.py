"""Phase 2 solvers for the expected-utility group:
compute_expected_utility, maximize_expected_utility, independence, independence_risk.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

NUM = r"-?\d+(?:,\d{3})*(?:\.\d+)?"


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def _out(index, pred, status) -> pd.DataFrame:
    return pd.DataFrame({"pred": pd.array(pred, dtype="Int64"), "status": status}, index=index)


def _match_numeric(options, value: float, tol: float):
    """Return (pred, status) matching a numeric answer against option texts (all numeric)."""
    vals = []
    for o in options:
        m = re.fullmatch(r"\s*\$?\s*(" + NUM + r")\s*", str(o))
        vals.append(_f(m.group(1)) if m else None)
    hits = [i for i, v in enumerate(vals) if v is not None and abs(v - value) <= tol]
    if len(hits) == 1:
        return hits[0], "ok"
    if len(hits) > 1:
        return None, "multi_match"
    return None, "no_match"


# ------------------------------------------------------------------------------ compute_expected_utility
_PCT = re.compile(r"(" + NUM + r")\s*%")
_MONEY = re.compile(r"\$\s*" + NUM)


def eu_from_text(q: str):
    """Expected utility stated in a compute_expected_utility question, or None if not parseable.

    Probabilities are the percentages in order; utilities are the remaining numbers (money amounts
    "$x" and percentages removed) in order. If there is one fewer probability than utilities, the last
    probability is the complement. Returns (eu, reason)."""
    probs = [_f(x) for x in _PCT.findall(q)]
    rest = _MONEY.sub(" ", _PCT.sub(" ", q))
    utils = [_f(x) for x in re.findall(r"(?<![\w.])" + NUM, rest)]
    if len(probs) == len(utils) - 1:
        probs.append(100 - sum(probs))
    if not probs or len(probs) != len(utils):
        return None, "unparsed"
    if abs(sum(probs) - 100) > 1e-6:
        return None, "ill_posed:probabilities_do_not_sum_to_1"
    return sum(p * u for p, u in zip(probs, utils)) / 100, "ok"


def solve_compute_eu(df: pd.DataFrame) -> pd.DataFrame:
    pred, status = [], []
    for q, opts in zip(df.question_text, df.options):
        eu, st = eu_from_text(q)
        if eu is None:
            pred.append(None); status.append(st); continue
        p, st = _match_numeric(list(opts), eu, tol=0.011)  # options are rounded to 2 dp
        pred.append(p); status.append(st)
    return _out(df.index, pred, status)


# ------------------------------------------------------------------------------ lottery parsing
_NONE_WORDS = re.compile(r"\b(?:no one|nobody|none|nothing|no prize|no people|no lives|zero)\b", re.I)
_NOISE = [  # numbers that are not outcomes or probabilities (die faces, wheel/generator labels, option numbers)
    re.compile(r"\b(?:lands? on|roll(?:s)?(?: an?)?|comes? up|numbers?)\s+(?:\d+\s*(?:-|to|,|or|and)?\s*)+", re.I),
    re.compile(r"\b(?:wheel|generator|door|box|option|plan|strategy|job|ticket|slot|machine|lottery|game|round|card)\s+[A-Z]?\d+\b", re.I),
    re.compile(r"\b[A-Z]\d+\b"),
    re.compile(r"\b\d+(?:st|nd|rd|th)\b"),
    re.compile(r"\b\d+\s*(?:years?|months?|weeks?|days?|hours?)\b", re.I),
]
_TOKEN = re.compile(
    r"(?P<pct>-?\d+(?:\.\d+)?)\s*%"
    r"|(?P<dprob>(?<=probability )0?\.\d+|(?<=probability of )0?\.\d+|(?<=chance of )0\.\d+)"
    r"|(?P<val>(?:-\s*\$\s*|\$\s*-?\s*|-)?\d+(?:,\d{3})*(?:\.\d+)?)")


def _clean(seg: str) -> str:
    s = _NONE_WORDS.sub(" 0 ", seg)
    for rx in _NOISE:
        s = rx.sub(lambda m: " " * len(m.group(0)), s)  # keep character positions
    return s


def _tokens(seg: str, pos: bool = False):
    s = _clean(seg)
    out = []
    for m in _TOKEN.finditer(s):
        if m.group("pct") is not None:
            t = ("P", float(m.group("pct")))
        elif m.group("dprob") is not None:
            t = ("P", 100 * float(m.group("dprob")))
        else:
            t = ("V", float(m.group("val").replace("$", "").replace(" ", "").replace(",", "")))
        out.append(t + (m.start(),) if pos else t)
    return out


def job_value(seg: str):
    """salary + p * bonus - extra living expenses, if the segment describes a job offer; else None.

    The bonus amount is the number closest to the word "bonus", the expense amount the number closest to
    "expenses"; the remaining number is the salary."""
    has_bonus = bool(re.search(r"\bbonus", seg, re.I))
    salaryish = bool(re.search(r"salary|income|\bpay|earn|per (?:year|annum)|annual", seg, re.I))
    if not (has_bonus or (salaryish and re.search(r"\badditional|\bextra", seg, re.I))) or not (salaryish or has_bonus):
        return None
    s = _NONE_WORDS.sub(" ", seg)
    toks = []
    for m in _TOKEN.finditer(s):
        if m.group("pct") is not None:
            toks.append(("P", float(m.group("pct")), m.start(), m.end()))
        elif m.group("val") is not None:
            toks.append(("V", float(m.group("val").replace("$", "").replace(" ", "").replace(",", "")),
                         m.start(), m.end()))
    P = [t for t in toks if t[0] == "P"]
    V = [t for t in toks if t[0] == "V"]
    def near(word, cands):
        pos = [m.start() for m in re.finditer(word, s, re.I)]
        if not pos or not cands:
            return None
        return min(cands, key=lambda t: min(min(abs(t[2] - q), abs(t[3] - q)) for q in pos))
    ew = r"expense|costs? of|commut"
    exp = near(ew, V) if re.search(ew, s, re.I) else None
    V2 = [t for t in V if t is not exp]
    if not P:
        if len(V2) == 1 and re.search(r"\bno\b|without|not? (?:any )?(?:chance|possibility)", s, re.I):
            return V2[0][1] - (exp[1] if exp else 0)
        return None
    if len(P) != 1 or len(V2) != 2:
        return None
    def dist(word, t):
        pos = [m.start() for m in re.finditer(word, s, re.I)]
        return min((min(abs(t[2] - q), abs(t[3] - q)) for q in pos), default=10 ** 6)
    bw = r"\bbonus" if has_bonus else r"\badditional|\bextra"
    sw = r"salary|income|\bpay\b|annually|per (?:year|annum)"
    sal = min(V2, key=lambda t: dist(sw, t) - dist(bw, t))
    bon = [t for t in V2 if t is not sal][0]
    return sal[1] + P[0][1] / 100 * bon[1] - (exp[1] if exp else 0)


def lottery(seg: str):
    """Parse one prospect description into [(prob in %, value)], or None."""
    t = _tokens(seg)
    P = [x for k, x in t if k == "P"]
    V = [x for k, x in t if k == "V"]
    kinds = "".join(k for k, _ in t)
    if not t:
        return None
    if not P:
        return [(100.0, V[0])] if len(V) == 1 else None
    if not V:
        return [(p, 1.0) for p in P] if len(P) == 1 else None  # "x% chance of success" (unit outcome)
    if len(P) == len(V):
        if len(P) == 1 and P[0] < 100:  # "x% chance of winning v" (nothing otherwise)
            return [(P[0], V[0]), (100 - P[0], 0.0)]
        return list(zip(P, V))
    if len(P) == len(V) - 1:  # last outcome is the complement ("..., otherwise you win y")
        return list(zip(P + [100 - sum(P)], V))
    if len(P) == len(V) + 1 and len(V) == 1 and kinds in ("PVP", "PPV"):
        return None
    return None


def ev(lot):
    if lot is None:
        return None
    tot = sum(p for p, _ in lot)
    if len(lot) > 1 and abs(tot - 100) > 0.05:
        return None
    return sum(p * v for p, v in lot) / 100


_COST = re.compile(r"\b(?:costs?|(?<!will )(?<!that )pay(?:ing)?(?= \$?\d+(?:\.\d+)? (?:to|for|upfront|per|as|and))|"
                   r"charges?|donation of|fee of|price of|buy-in of|entry of)\s+"
                   r"(?:a fee of\s+|an? \w+ fee of\s+)?(?:only\s+)?(\$\s*\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?)"
                   r"(?:\s*(?:dollars|per (?:play|spin|game|ticket|entry|round)))?", re.I)


def prospect_value(seg: str):
    j = job_value(seg)
    if j is not None:
        return j, "job"
    cost = 0.0
    m = _COST.search(seg)
    if m:
        cost = _f(m.group(1).replace("$", "").strip())
        seg = seg[:m.start()] + " " + seg[m.end():]
    lot = lottery(seg)
    if lot is not None and cost:
        if len(lot) == 1 and lot[0][1] == 1.0 and lot[0][0] < 100:
            return None, "unparsed"
        e = ev(lot)
        return (None, "unparsed") if e is None else (e - cost, "cost")
    if lot is None:
        return None, "unparsed"
    if len(lot) == 1 and lot[0][1] == 1.0 and lot[0][0] < 100:
        return lot[0][0] / 100, "success_prob"
    e = ev(lot)
    return e, ("ok" if e is not None else "unparsed:probs")


def _argmax(vals, lots_abs):
    """Unique maximiser. Values closer than max(0.011, 1e-4 * max|EV|) count as a tie: the generator rounds
    percentages to 2 decimals and often designs exactly equal expected values (e.g. 11 vs 11.0008)."""
    best = max(vals)
    tol = max(0.011, 1e-4 * max(lots_abs))
    hits = [i for i, v in enumerate(vals) if best - v <= tol]
    return (hits[0], "ok") if len(hits) == 1 else (None, "ill_posed:tie")


# ------------------------------------------------------------------------------ maximize_expected_utility
_LABEL = re.compile(
    r"\b(Job|Plan|Strategy|Option|Approach|Course|Treatment|Drug|Program|Programme|Generator|Patient|Investment|"
    r"Project|Route|Protocol|Method|Policy|Door|Box|Wheel|Game|Vaccine|Operation|Procedure|Company|Intervention|"
    r"Lottery|Ticket|Gamble|Offer|Prize|Plan)\s+([A-Z]\d?|\d)\b")
_CERTAIN = re.compile(r"\b(guarantee[ds]?|certain(?:ty|ly)?|sure|definitely|fixed|steady|stable|assured|"
                      r"for certain|no (?:chance|risk)|without|safe)\b", re.I)
_SAFE = re.compile(r"\b(safe|guarantee[ds]?|certain(?:ly)?|sure|conservative|proven|well-studied)\b", re.I)
_RISKY = re.compile(r"\b(risk(?:y|ier)?|gamble|experimental|chance|lottery|uncertain)\b", re.I)


def _label(opt: str):
    m = _LABEL.search(opt)
    return f"{m.group(1).lower()} {m.group(2)}" if m else None


def _label_segments(text: str) -> dict:
    """label -> list of text segments (from a mention to the next mention of a different label)."""
    ms = [(m.start(), f"{m.group(1).lower()} {m.group(2)}") for m in _LABEL.finditer(text)]
    segs: dict[str, list[str]] = {}
    i = 0
    while i < len(ms):
        j = i
        while j + 1 < len(ms) and ms[j + 1][1] == ms[i][1]:
            j += 1
        end = ms[j + 1][0] if j + 1 < len(ms) else len(text)
        segs.setdefault(ms[i][1], []).append(text[ms[i][0]:end])
        i = j + 1
    return segs


def _seg_value(segs, with_src=False):
    for s in segs or []:
        v, st = prospect_value(s)
        if v is not None:
            return (v, s) if with_src else v
    return (None, None) if with_src else None


def _tok_lottery(t):
    """Lottery from a token list [(kind, x)] if it is a sure amount or a complete gamble."""
    P = [x for k, x in t if k == "P"]
    V = [x for k, x in t if k == "V"]
    if not P and len(V) == 1:
        return [(100.0, V[0])]
    if len(P) >= 2 and len(P) == len(V) and abs(sum(P) - 100) <= 0.05:
        return list(zip(P, V))
    if len(P) >= 1 and len(V) == len(P) + 1 and sum(P) < 100:
        return list(zip(P + [100 - sum(P)], V))
    return None


def _split_two(text: str):
    """Values of the two prospects described in `text` (in text order), splitting its number sequence
    into exactly two complete prospects; None if no split or several different splits are possible."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    cands = []
    for k in range(1, len(sents)):
        a, b = " ".join(sents[:k]), " ".join(sents[k:])
        if _tokens(a) and _tokens(b):
            (va, ka), (vb, kb) = prospect_value(a), prospect_value(b)
            if va is not None and vb is not None:
                cands.append((va, vb, len(_tokens(a)) == 1, len(_tokens(b)) == 1))
    cands = list(dict.fromkeys(cands))
    if len(cands) == 1:
        return cands[0]
    tp = _tokens(text, pos=True)
    t = [x[:2] for x in tp]
    cert = [m.end() for m in _CERTAIN.finditer(text)]
    cands = []
    for k in range(1, len(t)):
        la, lb = _tok_lottery(t[:k]), _tok_lottery(t[k:])
        if la and lb:
            # a sure amount must be introduced by a certainty word shortly before it
            sure_pos = [tp[0][2]] if len(la) == 1 else [tp[k][2]] if len(lb) == 1 else []
            if sure_pos and not any(0 <= sure_pos[0] - c <= 80 for c in cert):
                continue
            cands.append((ev(la), ev(lb), len(la) == 1, len(lb) == 1))
    cands = list(dict.fromkeys(cands))
    return cands[0] if len(cands) == 1 else None


def _values_for_row(q: str, opts, prev_texts):
    """Expected value of every option, or None where it cannot be derived."""
    vals, sure, srcs = [], [], []
    segs_here = _label_segments(q)
    segs_prev = [_label_segments(t) for t in prev_texts]
    for o in opts:
        v, src = None, None
        lab = _label(o)
        if lab:  # a labelled prospect is described in the question text (this part first, then earlier parts)
            v, src = _seg_value(segs_here.get(lab), True)
            for sp in reversed(segs_prev):
                if v is None:
                    v, src = _seg_value(sp.get(lab), True)
        if v is None and re.search(r"\d", _NOISE[1].sub(" ", _NOISE[2].sub(" ", o))):
            v, kind = prospect_value(o)
            src = o
            if v is not None and kind == "ok" and not _PCT.search(o) and _PCT.search(q) \
                    and not _CERTAIN.search(o):
                v = None  # "may save 35": outcome without its probability; use the question text
        vals.append(v)
        sure.append(None if v is None else not _PCT.search(src))
        srcs.append(src)
    if len(opts) == 2 and vals[0] is None and vals[1] is None and not prev_texts:
        sp = _split_two(q)
        if sp is not None:
            ta, tb, sa, sb = sp
            safe = [bool(_SAFE.search(o)) and not _RISKY.search(o) for o in opts]
            risky = [bool(_RISKY.search(o)) and not _SAFE.search(o) for o in opts]
            if sa != sb and (safe[0] and risky[1] or safe[1] and risky[0]):
                sure_first = safe[0]
                vals, sure = ([ta, tb], [sa, sb]) if sa == sure_first else ([tb, ta], [sb, sa])
            else:
                # order-independent: place each option by the first mention in the text of a name it carries
                # ("Firm A", "Startup B"); ambiguous -> no value (was: text order == option order, which broke
                # once options are shuffled)
                pos = []
                for o in opts:
                    names = re.findall(r"\b[A-Z][a-z]+ [A-Z]\b", o)
                    hits = [q.find(n) for n in names if q.find(n) >= 0]
                    pos.append(min(hits) if hits else None)
                if None not in pos and pos[0] != pos[1]:
                    vals, sure = ([ta, tb], [sa, sb]) if pos[0] < pos[1] else ([tb, ta], [sb, sa])
    if prev_texts:
        # follow-up that splits ONE new gamble into several options ("C: 57.15% chance of saving 73, or
        # D: 42.85% chance of saving 80"): single-probability prospects whose probabilities sum to 100
        single = [[x for k, x in _tokens(sc) if k == "P"] for sc in srcs if sc is not None and sc in q]
        single = [p_[0] for p_ in single if len(p_) == 1]
        if len(single) >= 2 and abs(sum(single) - 100) <= 0.05:
            return "garbled", sure
    return vals, sure


_OLD = re.compile(r"\b(stick|original(?:ly)?|initial(?:ly)?|keep|continue|previous(?:ly)?|old|same|remain|stay|"
                  r"current|already|as planned|earlier|first)\b", re.I)
_NEW = re.compile(r"\b(new|switch|alternative|revised|updated|another|different|untested|third|instead|improved|"
                  r"modified|latest|novel)\b", re.I)


def _strip_letter(o: str) -> str:
    return re.sub(r"^\s*[A-H][.)]\s*", "", o).strip().rstrip(".").lower()


def _followup_values(q: str, opts, prev_q: str, prev_vals, prev_sure):
    """Part-1 fallback: one option is the earlier risky prospect, the other a new prospect in this text."""
    if len(opts) != 2 or prev_vals is None or any(v is None for v in prev_vals):
        return None
    gam = [v for v, s_ in zip(prev_vals, prev_sure) if not s_]
    if len(gam) != 1:
        return None
    ql, pl = q.lower(), prev_q.lower()
    role = []
    for o in opts:
        t = _strip_letter(o)
        if t and t in pl and t not in ql:
            role.append("old")
        elif t and t in ql and t not in pl:
            role.append("new")
        elif _OLD.search(o) and not _NEW.search(o):
            role.append("old")
        elif _NEW.search(o) and not _OLD.search(o):
            role.append("new")
        else:
            role.append(None)
    if sorted(role, key=str) != ["new", "old"]:
        return None
    new_v = job_value(q)
    if new_v is None:
        prev_nums = {x for _, x in _tokens(prev_q)}
        t = _tokens(q)
        while t and t[0][1] in prev_nums and _tok_lottery(t) is None:
            t = t[1:]
        lot = _tok_lottery(t)
        if lot is None or len(lot) < 2:
            return None
        new_v = ev(lot)
    return [gam[0] if r == "old" else new_v for r in role]


_LOWER = re.compile(r"\b(?:lower|smaller|reduced)(?: base| fixed| annual| starting)? (?:salary|pay)(?: of)? \$?(" + NUM + r")", re.I)
_SAL = re.compile(r"(?:salary|income|pay)(?: of)? \$?(" + NUM + r")|\$?(" + NUM + r")(?= (?:annually|per (?:year|annum)))", re.I)


def text_contradiction(q: str):
    """The question calls a salary 'lower' although it is the highest salary mentioned."""
    m = _LOWER.search(q)
    if not m:
        return False
    low = _f(m.group(1))
    others = [_f(a or b) for a, b in _SAL.findall(q)]
    others = [x for x in others if x != low]
    return bool(others) and low > max(others)


def solve_maximize_eu(df: pd.DataFrame) -> pd.DataFrame:
    pred = pd.array([pd.NA] * len(df), dtype="Int64")
    status = np.array(["unparsed"] * len(df), dtype=object)
    d = df.reset_index(drop=True)
    by_base = d.groupby("base_id", sort=False).indices
    texts = d.question_text.to_numpy()
    parts = d.part.to_numpy()
    for b, idx in by_base.items():
        idx = sorted(idx, key=lambda i: parts[i])
        prev, pv, ps = [], None, None
        for i in idx:
            opts = [str(o) for o in d.options.iat[i]]
            vals, sure = _values_for_row(texts[i], opts, prev)
            if isinstance(vals, str):
                status[i] = "ill_posed:garbled_followup"
                prev.append(texts[i])
                pv, ps = None, None
                continue
            if any(v is None for v in vals) and prev:
                fu = _followup_values(texts[i], opts, prev[-1], pv, ps)
                if fu is not None:
                    vals = fu
            prev.append(texts[i])
            pv, ps = vals, sure
            if len(opts) < 2 or any(v is None for v in vals):
                continue
            p, st = _argmax(vals, [abs(v) for v in vals])
            if text_contradiction(texts[i]):
                p, st = None, "ill_posed:text_contradicts_numbers"
            pred[i] = p if p is not None else pd.NA
            status[i] = st
    return _out(df.index, pred, status)


_LETTER = re.compile(r"^[A-H]$")
_FRAG = re.compile(r"^\$?-?[\d,.]+%?$")


def reassemble_options(opts):
    """Repair option lists that were exploded into fragments in the release; None if the list is fine.

    - characters:     ['A', '.', '', 'D', 'r', ...]        -> ['Drug X', 'Drug Y']
    - letter + text:  ['A', 'Receive $76', 'B', 'Gamble']   -> ['Receive $76', 'Gamble']
    - numeric pieces: ['Job E', '$112447', '42.89%', ...]   -> ['Job E', 'Job F']
    """
    o = [str(x) for x in opts]
    if len(o) >= 6 and all(len(x) <= 1 for x in o):
        s = "".join(x if x else " " for x in o)
        parts = [p.strip() for p in re.split(r"(?:^|\s)[A-H]\.\s*", s) if p.strip()]
        return parts if len(parts) >= 2 else None
    if len(o) >= 4 and len(o) % 2 == 0 and all(_LETTER.match(o[i]) for i in range(0, len(o), 2)) \
            and all(len(o[i]) > 2 for i in range(1, len(o), 2)):
        return [o[i] for i in range(1, len(o), 2)]
    if any(_FRAG.match(x) for x in o):
        keep = [re.sub(r"^[A-H]\.\s*", "", x) for x in o if not _FRAG.match(x) and not _LETTER.match(x)]
        if not keep:
            letters = [x for x in o if _LETTER.match(x)]
            keep = letters[1::2] if len(letters) == 4 else letters  # ['A', 'E', ..., 'B', 'F'] -> E, F
        keep = [x if x.lower().startswith("job") else f"Job {x}" for x in keep]
        return keep if len(keep) >= 2 and len(set(keep)) == len(keep) else None
    return None


def fix_maximize_eu(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new = df.options.map(reassemble_options)
    m = new.notna()
    df.loc[m, "options"] = new[m]
    # the stored flags pointed at fragments; the (authoritative) solver fills the answer
    df.loc[m, "correct"] = new[m].map(lambda o: [0.0] * len(o))
    df.loc[m, "option_ids"] = new[m].map(lambda o: list(range(len(o))))
    df.loc[m, "repair"] = df.loc[m, "repair"].map(lambda r: r + ["options_reassembled"])
    return df


# ============================================================================================ registries
REPAIRS = {
    "options_reassembled": "option list exploded into characters / letter+text / numeric fragments in the "
                           "release; reassembled into whole option texts (answer re-derived by the solver)",
}

SOLVERS = {
    "compute_expected_utility": solve_compute_eu,
    "maximize_expected_utility": solve_maximize_eu,
}

POLICY = {
    "compute_expected_utility": {"grading": "answer", "authoritative": True, "drop_status": ["ill_posed"],
                                 "shuffle": None},
    "maximize_expected_utility": {"grading": "answer", "authoritative": True, "drop_status": ["ill_posed"],
                                  "shuffle": None},
}

FIXES = {
    "maximize_expected_utility": fix_maximize_eu,
}


# ============================================================================================ independence_risk
_IR_PREF = [
    (re.compile(r"\bprefer(?:ence for|s|red)?\s+(?:having\s+|getting\s+)?(.+?)\s+(?:over|to|rather than|more than)\s+(.+?)(?:[.,;]|\s+and\b|$)", re.I), 0),
    (re.compile(r"between\s+(.+?)\s+and\s+(.+?),?\s+(?:you|I)\s+(?:would\s+|will\s+)?(?:choose|pick|select|prefer|go for|opt for)\s+(.+?)(?:[.,;]|$)", re.I), 1),
    (re.compile(r"\b(?:like|value|enjoy|favou?r)\s+(.+?)\s+more than\s+(.+?)(?:[.,;]|$)", re.I), 0),
    (re.compile(r"\brather have\s+(.+?)\s+than\s+(.+?)(?:[.,;]|$)", re.I), 0),
    (re.compile(r"(.+?)\s+is (?:preferred|favou?red)\s+(?:over|to)\s+(.+?)(?:[.,;]|$)", re.I), 0),
]
_IR_LABEL = re.compile(r"\bOption\s+([XY])\b")


def _ir_item(s: str) -> str:
    s = re.sub(r"^(?:an?|the)\s+", "", s.strip(), flags=re.I)
    return s.strip().lower()


def solve_independence_risk_one(q: str, options: list[str]):
    """Sure-thing principle: X and Y share the common consequence; the option whose differing outcome is the
    stated preferred item is the answer."""
    pref = None
    for rx, kind in _IR_PREF:
        m = rx.search(q)
        if not m:
            continue
        if kind == 0:
            pref = (_ir_item(m.group(1)), _ir_item(m.group(2)))
        else:
            a, b, ch = _ir_item(m.group(1)), _ir_item(m.group(2)), _ir_item(m.group(3))
            if a in ch and b not in ch:
                pref = (a, b)
            elif b in ch:
                pref = (b, a)
        if pref:
            end = m.end()
            break
    if not pref or not pref[0] or not pref[1] or pref[0] == pref[1]:
        return None, "unparsed"
    rest = q[end:].lower()
    pos_p = [m.start() for m in re.finditer(r"\b" + re.escape(pref[0]) + r"\b", rest)]
    pos_q = [m.start() for m in re.finditer(r"\b" + re.escape(pref[1]) + r"\b", rest)]
    # a longer item containing the other (e.g. "novel" in "mystery novel") is resolved by excluding overlaps
    if pref[1] in pref[0]:
        pos_q = [x for x in pos_q if not any(p <= x < p + len(pref[0]) for p in pos_p)]
    if pref[0] in pref[1]:
        pos_p = [x for x in pos_p if not any(p <= x < p + len(pref[1]) for p in pos_q)]
    if len(pos_p) != 1 or len(pos_q) != 1:
        return None, "unparsed"
    labels = []
    for m in _IR_LABEL.finditer(q[end:]):
        if m.group(1) not in labels:
            labels.append(m.group(1))
    if labels != ["X", "Y"] and labels != ["Y", "X"]:
        return None, "unparsed"
    want = labels[0] if pos_p[0] < pos_q[0] else labels[1]
    hits = [i for i, o in enumerate(options) if re.fullmatch(r"\s*Option\s+" + want + r"\s*\.?\s*", str(o))]
    if len(hits) == 1:
        return hits[0], "ok"
    return None, "multi_match" if hits else "no_match"


def solve_independence_risk(df: pd.DataFrame) -> pd.DataFrame:
    res = [solve_independence_risk_one(q, [str(x) for x in o]) for q, o in zip(df.question_text, df.options)]
    return _out(df.index, [r[0] for r in res], [r[1] for r in res])


SOLVERS["independence_risk"] = solve_independence_risk
POLICY["independence_risk"] = {"grading": "answer", "authoritative": True, "drop_status": ["ill_posed"],
                               "shuffle": None}


# ============================================================================================ independence
# Convention: consistency-graded. Part 0 offers Brand A / Brand B; part 1 adds Brand C (always dominated
# by A). Part 1 is consistent iff it picks the same brand as part 0; picking C is always inconsistent.
# Encoding (same convention as biases.py): consistency_group = "independence/<base_id>", consistency_rule =
# "same_brand:<i0><i1>,..." listing the allowed (part-0 index, part-1 index) pairs in the released (shuffled)
# options. POLICY.shuffle is False because the hook shuffles before writing the indices.
_BRAND = re.compile(r"^\s*Brand ([A-Z])\b")


def consistency_independence(df: pd.DataFrame) -> pd.DataFrame:
    from solvers.biases import _new_index, _shuffle_rows
    df, perms = _shuffle_rows(df, "independence")
    rule = pd.Series(None, index=df.index, dtype=object)
    grp = df.groupby("base_id").indices
    for b, ix in grp.items():
        if len(ix) != 2:
            continue
        rows = df.iloc[ix].sort_values("part")
        if list(rows.part) != [0, 1]:
            continue
        (i0, r0), (i1, r1) = rows.iterrows()
        lab0 = [(_BRAND.match(o) or [None, None])[1] for o in r0.options]
        lab1 = [(_BRAND.match(o) or [None, None])[1] for o in r1.options]
        if sorted(lab0) != ["A", "B"] or sorted(lab1) != ["A", "B", "C"]:
            continue
        pairs = sorted((lab0.index(x), lab1.index(x)) for x in ("A", "B"))
        r = "same_brand:" + ",".join(f"{i}{j}" for i, j in pairs)
        rule[i0] = r
        rule[i1] = r
    df["consistency_group"] = "independence/" + df.base_id
    df["consistency_rule"] = rule
    bad = df["drop"].isna() & df.consistency_rule.isna()
    df.loc[bad, "drop"] = "no_consistency_rule"
    return df


POLICY["independence"] = {"grading": "consistency", "authoritative": False, "drop_status": [], "shuffle": False}
CONSISTENCY = {"independence": consistency_independence}

"""Phase 2 solvers for the `axioms` group:
completeness, completeness_risk, transitivity, transitivity_risk, time_inconsistency, sunk_cost.

All six are answer-graded: each question part has exactly one correct option, derived here from the
question text and option texts only (never from option positions or stored answers).

  completeness(_risk)  two alternatives with a stated number (lives saved / return / percentage); the
                       completeness answer is the alternative with the larger number (strict preference).
  transitivity         stated pairwise preferences -> transitive closure -> the option that beats every other
                       option; undetermined / cyclic / contradictory statements are ill-posed.
  transitivity_risk    Preference X: L1 > L2, Preference Y: L2 > L3  =>  "L1 over L3".
  time_inconsistency   stationarity: the same (sooner-smaller / later-larger) choice as today-vs-tomorrow.
  sunk_cost            the option that ignores sunk costs (abandon / switch), checked against the stated
                       future cost and future value where the question gives both.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"

REPAIRS = {
    "option_label_stripped": "leaked '(Correct option)' / '(Incorrect option)' suffix removed from option text",
    "options_truncated": "options belonging to other questions (after an empty option / 'Example N:' "
                         "fragment) removed",
}


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def _out(index, pred, status) -> pd.DataFrame:
    return pd.DataFrame({"pred": pd.array(pred, dtype="Int64"), "status": status}, index=index)


# ============================================================================================ time_inconsistency
_TI_AMT = re.compile(r"\$?\s*(" + NUM + r")\s*(?:dollars\s+)?(today|tomorrow)")
_TI_CHOSE = re.compile(r"(?:chose(?:n)?|choice) (?:to receive )?\$?\s*(" + NUM + r")\s*(?:dollars\s+)?(today|tomorrow)")
_TI_OPT = re.compile(r"^\s*(" + NUM + r") dollars (\d+) days from now\s*$")


def solve_time_inconsistency_one(q: str, options: list[str]):
    """Stationarity: choose the same relative option (sooner-smaller vs later-larger) as in today/tomorrow."""
    ch = _TI_CHOSE.search(q)
    if not ch:
        return None, "unparsed"
    chosen_amt, chosen_when = _f(ch.group(1)), ch.group(2)
    amts = {}
    for m in _TI_AMT.finditer(q[: ch.start()] + " " + q[ch.end():]):
        amts.setdefault(m.group(2), _f(m.group(1)))
    if set(amts) != {"today", "tomorrow"}:
        # "chose to receive $X today rather than $Y tomorrow": the other amount follows the chosen one
        amts[chosen_when] = chosen_amt
        if set(amts) != {"today", "tomorrow"}:
            return None, "unparsed"
    if amts[chosen_when] != chosen_amt:
        return None, "ill_posed:chosen_amount_not_offered"
    if not amts["tomorrow"] > amts["today"]:
        return None, "ill_posed:later_amount_not_larger"
    parsed = [_TI_OPT.match(o) for o in options]
    if not all(parsed) or len(options) != 2:
        return None, "unparsed"
    vals = [(_f(m.group(1)), int(m.group(2))) for m in parsed]
    (a0, d0), (a1, d1) = sorted(vals, key=lambda v: v[1])
    if d1 - d0 != 1:
        return None, "ill_posed:delay_gap_not_one_day"
    if (a0, a1) != (amts["today"], amts["tomorrow"]):
        return None, "ill_posed:amounts_differ_from_first_choice"
    want = (a0, d0) if chosen_when == "today" else (a1, d1)
    return vals.index(want), "ok"


def solve_time_inconsistency(df: pd.DataFrame) -> pd.DataFrame:
    res = [solve_time_inconsistency_one(q, list(o)) for q, o in zip(df.question_text, df.options)]
    return _out(df.index, [r[0] for r in res], [r[1] for r in res])


# ============================================================================================ transitivity_risk
_TR_PREF = re.compile(r"Preference ([XY]): (.+?) is preferred over (.+?)\.?\s*$", re.M)
_TR_OPT = re.compile(r"^(.+?) over (.+?)\s*$")


def _lot(s: str) -> str:
    s = s.strip().rstrip(".")
    s = re.sub(r"^(?:an?|A|An)\s+", "", s)
    return s.lower()


def solve_transitivity_risk_one(q: str, options: list[str]):
    prefs = _TR_PREF.findall(q)
    if len(prefs) != 2:
        return None, "unparsed"
    edges = [(_lot(a), _lot(b)) for _, a, b in prefs]
    (a, b), (c, d) = edges
    if len({a, b, c, d}) != 3:
        return None, "ill_posed:not_a_chain"
    if b == c:
        top, bottom = a, d
    elif d == a:
        top, bottom = c, b
    else:
        return None, "ill_posed:not_a_chain"
    hits, cant = [], []
    for i, o in enumerate(options):
        m = _TR_OPT.match(o)
        if m and (_lot(m.group(1)), _lot(m.group(2))) == (top, bottom):
            hits.append(i)
        if re.search(r"cannot determine", o, re.I):
            cant.append(i)
    if len(hits) == 1:
        return hits[0], "ok"
    return None, "multi_match" if hits else "no_match"


def solve_transitivity_risk(df: pd.DataFrame) -> pd.DataFrame:
    res = [solve_transitivity_risk_one(q, list(o)) for q, o in zip(df.question_text, df.options)]
    return _out(df.index, [r[0] for r in res], [r[1] for r in res])



# ============================================================================================ completeness(_risk)
_PLACEHOLDER = re.compile(r"\{[^}]*\}|\b(?:num_victims|save_[ab]|return_[ab]|percent_(?:[ab]|current))\b")
_STOP = set("""a an the of to in on for with and or but by as at is are be it its this that which who will would
can could you your i my me we our they their them he she his her from than then so if not no do does
has have had was were more most less option options choose choice prefer preferred preference select opt
implement use go""".split())
_NONCHOICE = re.compile(
    r"indifferen|equal|either|no preference|not have a preference|both|neither|cannot|can't|can not|unable|"
    r"impossible|need (?:more|additional|further)|require|without|additional (?:data|information)|"
    r"further (?:data|information)|more (?:data|information)|wait|postpone|defer|depends|split|portion|"
    r"random|seek|undecided|not possible|none|third|would not|makes no difference|no difference|same|"
    r"not enough|consult|evenly|combination|matter|abstain|insufficient|unsure|relies|rely on|diversify|"
    r"irrespective|mattress|hold cash", re.I)
# templates whose two alternatives are not comparable on one number, or where one value is missing
_PAIR_ILL = [
    (re.compile(r"uncertain how many", re.I), "ill_posed:missing_value"),
    (re.compile(r"rehabilitation uncertain", re.I), "ill_posed:incomparable_outcomes"),
    (re.compile(r"reduce infection rates by .* reduce mortality by", re.I), "ill_posed:incomparable_outcomes"),
    (re.compile(r"prefer the investment offering", re.I), "ill_posed:alternatives_not_labelled"),
]
_ORDER = {"A": 0, "B": 1, "1": 0, "2": 1, "X": 0, "Y": 1}
_LABEL = re.compile(r"(?<![\w$])(?:[A-HJ-Z]|[12]|X_\d|Alpha|Beta|One|Two)(?![\w%])")


def _tok(s: str) -> list[str]:
    out = []
    for t in re.findall(r"[A-Za-z]+(?:_\d+)?|\d+", s):
        out.append(t if (len(t) == 1 and t.isupper()) or "_" in t else t.lower())
    return out


def _stem(t: str) -> str:
    return t[:5] if len(t) > 5 else t


def _values(text: str, kind: str):
    """(position, value) of the compared quantities in `text`."""
    if kind == "percent":
        rx = re.compile(r"(" + NUM + r")\s*%")
    elif kind == "money":
        rx = re.compile(r"\$\s*(" + NUM + r")")
    else:  # counts (lives): plain numbers that are not labels such as "Protocol 1" / "X_1"
        rx = re.compile(r"(?<![\w$.,])(" + NUM + r")(?![\w%]|,\d|\.\d)")
    vals = []
    for m in rx.finditer(text):
        if kind == "count":
            pre = text[: m.start()].rstrip()
            if re.search(r"(?:\b[A-Z][a-z]+|_)$", pre) and _f(m.group(1)) < 10 and not re.search(
                    r"(?:save|saving|treat|help|rescue|cure|immunize)\s*$", pre, re.I):
                continue  # "Protocol 1", "Option 2"
        vals.append((m.start(), _f(m.group(1)), m.end()))
    return vals


def _occurrences(opt: str, other: str, q: str) -> list[tuple[int, int]]:
    """Spans in q of the tokens that distinguish option `opt` from option `other`."""
    ot = {_stem(t) for t in _tok(other)}
    toks = {_stem(t) for t in _tok(opt) if t not in _STOP and (not t.isdigit() or len(t) == 1)} - ot
    labels = {t for t in toks if len(t) == 1 or "_" in t}
    olabels = {t for t in ({_stem(t) for t in _tok(other)} - toks) if len(t) == 1 or "_" in t}
    if labels and olabels:  # both alternatives are labelled ("Plan A" / "Plan B"): use the labels only
        toks = labels
    else:  # "Investment A because ... than Investment B": the first label names the option
        la = [m.group(0) for m in _LABEL.finditer(opt)]
        lb = [m.group(0) for m in _LABEL.finditer(other)]
        if la and lb and la[0] != lb[0] and set(la) == set(lb):
            toks = {la[0]}
    spans = []
    for m in re.finditer(r"[A-Za-z]+(?:_\d+)?|\d+", q):
        t = m.group(0)
        t = t if (len(t) == 1 and t.isupper()) or "_" in t else t.lower()
        if _stem(t) not in toks:
            continue
        if t == "A" and re.match(r"\s+[a-z]", q[m.end():]) and re.search(r"(?:^|[.!?:]\s*)$", q[: m.start()]):
            continue  # sentence-initial article "A ..."
        spans.append((m.start(), m.end()))
    return spans


def _dist(spans, num_span) -> float:
    s0, e0 = num_span
    d = [max(s0 - e, st - e0, 0) for st, e in spans]
    return min(d) if d else float("inf")


def solve_pair_one(q: str, options: list[str], kind: str, drop_max: bool):
    """Larger-is-better choice between two described alternatives."""
    if _PLACEHOLDER.search(q):
        return None, "ill_posed:unfilled_placeholder"
    for rx, st in _PAIR_ILL:
        if rx.search(q):
            return None, st
    if any(_PLACEHOLDER.search(o) for o in options):
        return None, "ill_posed:unfilled_placeholder_in_options"
    choice = [i for i, o in enumerate(options) if not _NONCHOICE.search(o)]
    if len(choice) != 2:
        # options with explanations ("Investment B, because even without ...") contain non-choice words;
        # fall back to options carrying a single distinct label that also appears in the question
        lab = []
        for i, o in enumerate(options):
            ls = {m.group(0) for m in _LABEL.finditer(o)}
            if len(ls) == 1 and re.search(r"(?<=[A-Za-z] )" + re.escape(next(iter(ls))) + r"(?![\w%])", q):
                lab.append(i)
        if len(lab) == 2 and len({next(iter({m.group(0) for m in _LABEL.finditer(options[i])})) for i in lab}) == 2:
            choice = lab
        else:
            return None, "unparsed:choice_options"
    i, j = choice
    ov = [_values(options[k], kind) for k in (i, j)]
    if all(len(x) == 1 for x in ov):
        vi, vj = ov[0][0][1], ov[1][0][1]  # both options state their own value
        ci, cj = options[i], options[j]
    else:
        qv = _values(q, kind)
        if drop_max and len(qv) >= 3:  # the number of victims (>= save_a + save_b by construction)
            mx = max(v for _, v, _ in qv)
            rest = [x for x in qv if x[1] != mx]
            if len(rest) >= 2 and mx >= rest[0][1] + rest[1][1]:
                qv = rest
        if kind == "percent" and len(qv) >= 3:
            # completeness_risk states the status quo first ("currently has a 14.26% chance"); it is not one of
            # the two alternatives (added in autosteer, October 2026: the solver compared it with option A)
            # Read: a value whose clause names the status quo; otherwise, with exactly three values, the first one
            # (the questions state the status quo before the two alternatives).
            rest = [x for x in qv if not _BASELINE.search(_clause(q, x[0]))]
            if len(rest) == 2:
                qv = rest
            elif len(qv) == 3:
                qv = qv[1:]
        if len(qv) < 2:
            return None, "ill_posed:missing_value"
        (p1, v1, e1), (p2, v2, e2) = qv[0], qv[1]
        oi, oj = _occurrences(options[i], options[j], q), _occurrences(options[j], options[i], q)
        if not oi and not oj:  # the question does not name the alternatives the options refer to
            la = [m.group(0) for m in _LABEL.finditer(options[i])]
            lb = [m.group(0) for m in _LABEL.finditer(options[j])]
            if re.search(r"\bthe first\b.*\bthe second\b", q) and la and lb and \
                    {_ORDER.get(la[0]), _ORDER.get(lb[0])} == {0, 1}:
                first_is_i = _ORDER[la[0]] == 0
            elif re.search(r"\bone\b.*\bthe other\b", q, re.I):
                return None, "ill_posed:alternatives_not_labelled"
            else:
                return None, "unparsed:anchor"
        else:
            straight = _dist(oi, (p1, e1)) + _dist(oj, (p2, e2))
            swapped = _dist(oi, (p2, e2)) + _dist(oj, (p1, e1))
            if straight == swapped or min(straight, swapped) > 300:
                return None, "unparsed:anchor"
            first_is_i = straight < swapped
        c1, c2 = _clause(q, p1), _clause(q, p2)
        vi, vj, ci, cj = (v1, v2, c1, c2) if first_is_i else (v2, v1, c2, c1)
    if vi == vj:
        return None, "ill_posed:tie"
    hi, lo = (ci, cj) if vi > vj else (cj, ci)
    if _UNCERTAIN.search(hi) and _CERTAIN.search(lo) and not _UNCERTAIN.search(lo):
        # a sure smaller outcome against an explicitly uncertain larger one: risk attitude decides
        return None, "ill_posed:larger_outcome_uncertain"
    return (i if vi > vj else j), "ok"


_BASELINE = re.compile(r"current|presently|present\b|existing|standard care|status quo|right now|at the moment|"
                       r"\bnow\b|baseline|already", re.I)
_UNCERTAIN = re.compile(r"potential|could|might|possib|chance|risk|uncertain|not certain|lower certainty|"
                        r"\bmay\b|gamble|volatil|speculat|probab|likelihood", re.I)
_CERTAIN = re.compile(r"guarantee|assur|\bcertain|definite|ensur|\bsure\b|known to|fixed|secure", re.I)
_CLAUSE_SPLIT = re.compile(r"[.;:?!]|,?\s+(?:while|whereas|but|or|and|on the other hand)\s+|,\s+")


def _clause(q: str, p: int) -> str:
    """The clause of q around character position p (context of one alternative's value)."""
    starts = [m.end() for m in _CLAUSE_SPLIT.finditer(q, 0, p)]
    nxt = _CLAUSE_SPLIT.search(q, p)
    # a leading "Treatment B, which ..." belongs to the same alternative: extend one clause back
    st = starts[-2] if len(starts) >= 2 and re.match(r"\s*which\b", q[starts[-1]:]) else (starts[-1] if starts else 0)
    return q[st: nxt.start() if nxt else len(q)]


def _solve_pair(kind_of, drop_max_of):
    def solve(df: pd.DataFrame) -> pd.DataFrame:
        res = [solve_pair_one(q, list(o), kind_of(d), drop_max_of(d))
               for q, o, d in zip(df.question_text, df.options, df.domain)]
        return _out(df.index, [r[0] for r in res], [r[1] for r in res])
    return solve


solve_completeness = _solve_pair(lambda d: "money" if d == "finance" else "count", lambda d: d != "finance")
solve_completeness_risk = _solve_pair(lambda d: "percent", lambda d: False)



# ============================================================================================ sunk_cost
_SC_PLACEHOLDER = re.compile(r"[{}]")
_SC_SUNK = re.compile(
    r"\b(?:continu\w*|keep\w*|stick\w*|stay\w*|complet\w*|finish\w*|proceed\w*|persist\w*|go ahead|go on|"
    r"carry on|pour\w*|inject\w*|additional|another|extra|remaining|already|sunk|despite|recover\w*|"
    r"commit\w*|push\w*|maintain\w*|remain\w*|resum\w*|further|more|same|current\w*|original\w*|old|"
    r"existing|previous\w*|initial\w*|so far|invested|spent|paid|bought|purchased|put in|pump\w*|"
    r"follow through|see it through|persever\w*|hold on|renew\w*)\b", re.I)
_SC_EXIT = re.compile(
    r"\b(?:abandon\w*|scrap\w*|switch\w*|stop\w*|drop\w*|cut\w*|quit\w*|try\w*|tried|begin\w*|start\w*|"
    r"transition\w*|cancel\w*|discontinu\w*|new\w*|halt\w*|pull\w*|end|ending|terminat\w*|sell\w*|shut\w*|"
    r"walk away|move\w*|opt\w*|adopt\w*|chang\w*|give up|giv\w* up|leav\w*|clos\w*|ceas\w*|ditch\w*|"
    r"write off|writ\w* off|let go|accept\w*|choos\w*|join\w*|purchas\w* the|buy\w*|go back|replac\w*|"
    r"explor\w*|seek\w*|transfer\w*|alternative|instead|suspend\w*|shelv\w*|redirect\w*|divest\w*|"
    r"sign up|enrol\w*|consult\w*|visit\w*|take up|pursu\w*|embrac\w*|implement\w*|utiliz\w*|"
    r"forgo|forego|withdraw\w*|exit\w*|dump\w*|trash\w*|nix\w*|kill\w*|liquidat\w*|salvag\w*|"
    r"reinvest\w*|re-invest\w*|redirect\w*|reallocat\w*|put an end|call off|back out|bow out)\b", re.I)
_SC_AMT = re.compile(r"\$?\s*(" + NUM + r")")
_SC_DROP = re.compile(
    r"\b(?:fall(?:s|en)?|fell|drop(?:s|ped)?|down|reduc\w*|decreas\w*|declin\w*|plummet\w*|lower\w*|shr[iu]nk\w*|"
    r"dwindl\w*|revis\w*|slump\w*|diminish\w*|tumbl\w*|plung\w*|slid\w*|dipp\w*|dip|"
    r"now (?:only )?(?:worth|valued at|expected|estimated|projected)|only)\b[^$.]{0,40}\$\s*(" + NUM + r")"
    r"(?:[^$.]{0,25}?\bto\s*\$\s*(" + NUM + r"))?", re.I)   # "... fallen from $415 million to $210 million"
# templates whose rational answer is not determined by the text (numbers garbled or benefit not quantified)
_SC_ILL = [
    (re.compile(r"your house is now worth .* buy a new house|Sell the house and buy a new house", re.I),
     "ill_posed:not_determined"),
    (re.compile(r"new and improved software of the same price became available", re.I), "ill_posed:not_determined"),
    (re.compile(r"expecting it to bring productivity growth of", re.I), "ill_posed:not_determined"),
    (re.compile(r"invest an additional \$[\d,]+ for upgrades hoping to increase the sale price", re.I),
     "ill_posed:not_determined"),
    (re.compile(r"rebuilding your app to match theirs", re.I), "ill_posed:not_determined"),
]
_SC_TEXT_COST = re.compile(
    r"\b(?:requir\w*|need\w*|cost\w*|take|takes|necessary|required)\b[^$.]{0,30}?(?:additional|extra|further|"
    r"another)\s*\$\s*(" + NUM + r")|(?:additional|extra|further|another)\s*\$\s*(" + NUM + r")[^.]{0,30}?"
    r"\b(?:is|are|would be|will be)?\s*(?:requir\w*|need\w*|necessary)|\b(?:requir\w*|need\w*)\s+(?:an? )?"
    r"(?:additional|extra|further|another)?\s*\$\s*(" + NUM + r")[^.]{0,20}?\b(?:to|more)\b|"
    r"(?:further|additional|extra|another) (?:investment|funding|capital|expenditure|spending|cost)s? of \$\s*("
    + NUM + r")|\$\s*(" + NUM + r")\s*(?:million |billion )?(?:more|additional|extra)\b", re.I)
_SC_BUDGET = re.compile(r"agreed expenditure of \$\s*(" + NUM + r")\.\s*\$\s*(" + NUM + r")[^.]*already been spent"
                        r"[^$]*fallen to \$\s*(" + NUM + r")", re.I)


_SC_LEAD_CONT = re.compile(
    r"^\W*(?:to |i would |)?(?:(?:you |he |she |they |i |we )?(?:would |should |will )?)?(?:continu\w*|keep\w*|stick\w*|stay\w*|"
    r"persist\w*|proceed\w*|complet\w*|finish\w*|go ahead|go on|carry on|pour\w*|inject\w*|hold\w*|"
    r"remain\w*|follow\w* through|maintain\w*|persever\w*|push\w* (?:on|forward|ahead)|commit\w*|resum\w*|"
    r"(?:invest|spend|put|allocate|pay|fund|provide|pull|add|pump|commit|inject)\w*(?: in)? (?:an? )?(?:another|"
    r"additional|extra|more|further|the (?:additional|extra|remaining|further)|\$\s*[\d,.]+\s*"
    r"(?:million |billion )?more)|buy\w* more|pay\w* \$|see it through|attend\w* the remaining|renew\w*|"
    r"repair\w* the|fix\w* the|upgrad\w* the (?:old|current|existing)|move\w* (?:forward|ahead)|"
    r"(?:pursu\w*|go\w* with|us\w*|opt\w* for|choos\w*|tak\w*|insist\w*)\b[^.,]{0,25}\b(?:current|original|"
    r"existing|initial|old|same|first|already|advanced dental procedure E|strict dietary plan)|insist\w*)\b", re.I)
_SC_LEAD_EXIT = re.compile(
    r"^\W*(?:to |i would |)?(?:(?:you |he |she |they |i |we )?(?:would |should |will )?)?(?:abandon\w*|scrap\w*|switch\w*|stop\w*|"
    r"drop\w*|cut\w*|quit\w*|try\w*|begin\w*|start\w*|transition\w*|cancel\w*|discontinu\w*|halt\w*|"
    r"pull\w*|end\w*|terminat\w*|sell\w*|shut\w*|walk\w*|move\w* (?:to|on)|opt\w*|adopt\w*|chang\w*|giv\w*|"
    r"leav\w*|clos\w*|ceas\w*|ditch\w*|writ\w*|let\w*|accept\w*|choos\w*|join\w*|buy\w*|purchas\w*|"
    r"go back|replac\w*|explor\w*|seek\w*|transfer\w*|take\w*|revert\w*|return\w*|sign\w* up|enrol\w*|"
    r"consult\w*|visit\w*|embrac\w*|implement\w*|forgo\w*|forego\w*|withdraw\w*|exit\w*|"
    r"dump\w*|trash\w*|liquidat\w*|suspend\w*|shelv\w*|redirect\w*|divest\w*|reallocat\w*|call\w* off|"
    r"back\w* out|abort\w*|refus\w*|decid\w* against|not (?:invest|spend|continu|proceed|complet)|"
    r"don't|do not|avoid\w*|reject\w*|declin\w*|gradually|look\w* for|go with the (?:new|other)|"
    r"use the (?:new|other|free|cheaper|alternative)|invest\w* in (?:the |a )?(?:new|other|ready|better|"
    r"more|alternative|cheaper|different))\b", re.I)


def _sc_side(o: str) -> int:
    """>0: option honours the sunk cost (continue); <0: option walks away / switches; 0 unknown."""
    lc = bool(_SC_LEAD_CONT.search(o))
    le = bool(_SC_LEAD_EXIT.search(o)) and not lc
    if lc != le:
        return 2 if lc else -2
    sc = len(_SC_SUNK.findall(o)) - len(_SC_EXIT.findall(o))
    return int(np.sign(sc))


def solve_sunk_cost_one(q: str, options: list[str]):
    if len(options) != 2:
        return None, "unparsed"
    if _SC_PLACEHOLDER.search(q) or any(_SC_PLACEHOLDER.search(o) for o in options):
        return None, "ill_posed:unfilled_placeholder"
    for rx, st in _SC_ILL:
        if rx.search(q) or any(rx.search(o) for o in options):
            return None, st
    # Convention: drop questions whose options carry amounts the question never states (mis-substituted
    # template slots, e.g. 'Buy the new one for $785' for a $466 oven)
    q_amts = {_f(x) for x in re.findall(r"\$\s*(" + NUM + r")", q)}
    for o in options:
        o_amts = {_f(x) for x in re.findall(r"\$\s*(" + NUM + r")", o)}
        if o_amts - q_amts:
            return None, "ill_posed:substituted_amount"
    s0, s1 = _sc_side(options[0]), _sc_side(options[1])
    if s0 == s1 or (abs(s0) < 2 and abs(s1) < 2 and s0 * s1 >= 0):
        return None, "unparsed:option_roles"
    cont, walk = (0, 1) if s0 > s1 else (1, 0)
    # remaining sessions / months of the current course
    m = re.search(r"\bremaining (" + NUM + r")\b|\bfor (?:another|the remaining) (" + NUM + r")\b", options[cont])
    if m and _f(m.group(1) or m.group(2)) == 0:
        return None, "ill_posed:nothing_remaining"
    b = _SC_BUDGET.search(q)  # "agreed expenditure of $E. $S million has already been spent ... fallen to $V"
    if b:
        remaining, value = _f(b.group(1)) - _f(b.group(2)), _f(b.group(3))
        if value == remaining:
            return None, "ill_posed:cost_equals_value"
        return (cont if value > remaining else walk), "ok"
    # projection templates: future cost of continuing vs revised future value
    amts = [_f(x) for x in re.findall(r"\$\s*(" + NUM + r")", options[cont])]
    if not amts:
        amts = [_f(x) for x in re.findall(r"(?:additional|another|extra|more)\s*\$?\s*(" + NUM + r")",
                                          options[cont])]
    if re.search(r"additional|another|extra|more|further|inject|pour|put in|commit", options[cont], re.I) and \
            len(amts) == 1:
        d = _SC_DROP.search(q)
        if d:
            cost, value = amts[0], _f(d.group(2) or d.group(1))
            tc = _SC_TEXT_COST.search(q)
            if tc:
                text_cost = _f(next(g for g in tc.groups() if g))
                if text_cost != cost:
                    # the option's amount contradicts the cost stated in the question (mis-filled template)
                    return None, "ill_posed:inconsistent_amounts"
            if value > cost:
                return cont, "ok"  # continuing is profitable: ignoring sunk costs means continuing
            if value == cost:
                return None, "ill_posed:cost_equals_value"
    return walk, "ok"


def solve_sunk_cost(df: pd.DataFrame) -> pd.DataFrame:
    cache: dict = {}
    pred, status = [], []
    for q, o in zip(df.question_text, df.options):
        k = (q, tuple(o))
        if k not in cache:
            cache[k] = solve_sunk_cost_one(q, list(o))
        p_, s_ = cache[k]
        pred.append(p_)
        status.append(s_)
    return _out(df.index, pred, status)


# ============================================================================================ transitivity
_TV_LABEL = re.compile(r"\s*\((?:in)?correct option\)\s*\.?\s*$", re.I)
_TV_STOP = _STOP | set("""also think believe like likes liked enjoy enjoys love rank ranks ranked rate rates find finds
consider considers much far even really just still would do does did stronger strong higher greater better best
over rather instead lot whereas while however on list feel want actually generally""".split())
_TV_NEG = re.compile(r"\b(?:less|lower|worse|fewer|smaller|weaker|below|inferior|not as|dislike|hate|least)\b", re.I)
_TV_PIVOT = [re.compile(r"\brather than\b|\binstead of\b|\bthan\b", re.I),
             re.compile(r"\bover\b|\babove\b|\bahead of\b|\bexceeds?\b|\bsurpass(?:es)?\b|\boutweighs?\b|\btrumps?\b|"
                        r"\bbeats?\b|\boutranks?\b", re.I),
             re.compile(r"\b(?:superior|preferable|preferred) to\b", re.I)]
_TV_NOTAS = re.compile(r"(?:\bnot|n't)\s+(?:quite\s+|nearly\s+)?as\b.*?\b(as)\b", re.I)
_TV_PREF_TO = re.compile(r"\bprefer(?:s|red)?\b.*\bto\b", re.I)
_TV_BETWEEN = re.compile(r"(?:(?:choice|choose|option|pick|decide|deciding|choosing)\s+)?between (.+?) (?:and|or) (.+?),?\s+"
                         r"(?:you|I|they)(?: would| will| always)? (?:choose|pick|select|prefer|go with|opt for|take|eat|"
                         r"pick|go for|favou?r)s? (.+?)(?=[.,;]|$)", re.I)
_TV_ATTR = re.compile(r"\b(?:more|less)\s+([a-z]+)\s+(?:than|over)\b|\b([a-z]+er)\s+than\b", re.I)
_TV_ATTR_SKIP = {"rather", "better", "worse", "higher", "lower", "greater", "stronger", "further", "other", "prefer",
                 "favor", "favour", "enjoy", "like", "love", "appreciate", "value", "trust", "admire", "want"}
_TV_SENT = re.compile(r"(?<=[.;!?:])\s+")
_TV_CHUNK = re.compile(r",?\s+(?:and|but|while|whereas|yet|however|although|though)\s+|,\s+|;\s*")
_TV_QUESTION = re.compile(r"(?:which|who|whom|what|where|choose|select|pick|decide on|name)\b", re.I)
_TV_QSTART = re.compile(r"(?:^|,\s*|\s)(?=(?:which|who|whom|what|where|how|would you|do you|can you|choose|select)\b)", re.I)
_TV_BOTTOM = re.compile(r"\bnot\b|n't\b|\bworst|\bleast\b|\blast (?:choice|place|option|pick|resort)|\b(?:choose|pick|rank|select)\w* last|"
                        r"\bleave\b.*\bout\b|\bleft out|\bsmallest|\bmost un\w+|\bbench|\bavoid|\beliminat|\bexclud|\bdrop|\blowest|"
                        r"\bbottom|\bweakest|\bless\b|\bfewest|\bleave out|\bskip|\breject|\bdislike|\bpoorest|\bwithout", re.I)


def _tv_attr(a: str) -> str:
    a = a.lower()
    if a.endswith("ier"):
        return a[:-3] + "y"
    if a.endswith("er") and len(a) > 5:
        a = a[:-2]
        return a[:-1] if len(a) > 2 and a[-1] == a[-2] else a
    return a


def _tv_norm(s: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", s.lower())) + " "


def _tv_sing(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def clean_transitivity_options(opts):
    """(options, n_label_stripped, truncated): leaked '(incorrect Option)' labels removed; options after the
    first empty option (fragments of the next few-shot example, e.g. 'mple 10:') dropped."""
    o = [str(x) for x in opts]
    trunc = False
    if "" in o and o.index("") >= 2:
        o, trunc = o[:o.index("")], True
    new = [_TV_LABEL.sub("", x).strip() for x in o]
    return new, sum(a != b for a, b in zip(o, new)), trunc


def _tv_side_entity(side: str, opts_n):
    """Option index whose text occurs in `side` (the occurrence nearest to `near_end`) or None."""
    sn = " " + " ".join(_tv_sing(t) for t in _tv_norm(side).split()) + " "
    best = None
    for i, on in enumerate(opts_n):
        k = sn.rfind(on)
        if k >= 0 and (best is None or len(on) > len(opts_n[best])):
            best = i
    if best is None:  # "Horse riding" vs "riding horses": every content token of the option, any order
        st = set(sn.split())
        hits = [i for i, on in enumerate(opts_n) if len(on.split()) > 1 and set(on.split()) - _TV_STOP <= st]
        best = hits[0] if len(hits) == 1 else None
    return best


def solve_transitivity_one(q: str, options: list[str]):
    opts = [str(o).strip() for o in options if str(o).strip()]
    if len(opts) < 2:
        return None, "ill_posed:fewer_than_two_options"
    opts_n = [" " + " ".join(_tv_sing(t) for t in _tv_norm(o).split()) + " " for o in opts]
    if len(set(opts_n)) != len(opts_n):
        return None, "ill_posed:duplicate_options"
    rel = []  # (winner_side_text, loser_side_text)
    attrs = set()
    bottom = False
    for sent in _TV_SENT.split(q.strip()):
        sent = sent.strip()
        if not sent:
            continue
        if _TV_QUESTION.match(sent):
            bottom = bottom or bool(_TV_BOTTOM.search(sent))
            continue
        if sent.endswith("?") or sent.endswith(":"):
            # "If you prefer X over Y, and like Z more than X, what's your go-to fruit?"
            qs = [m.start() for m in _TV_QSTART.finditer(sent)]
            k = qs[-1] if qs else 0
            bottom = bottom or bool(_TV_BOTTOM.search(sent[k:]))
            sent = re.sub(r"^(?:if|suppose|given that|assuming)\s+", "", sent[:k], flags=re.I)
            if not sent.strip():
                continue
        for m in _TV_BETWEEN.finditer(sent):
            a, b, ch = m.group(1), m.group(2), m.group(3)
            chn, an, bn = _tv_norm(ch), _tv_norm(a), _tv_norm(b)
            if re.search(r"\bformer\b", ch, re.I):
                rel.append((a, b))
            elif re.search(r"\blatter\b", ch, re.I):
                rel.append((b, a))
            elif an.strip() and an in chn and not (bn.strip() and bn in chn):
                rel.append((a, b))
            elif bn.strip() and bn in chn:
                rel.append((b, a))
        sent = _TV_BETWEEN.sub(" ; ", sent)
        prev_piv = None
        for c in _TV_CHUNK.split(sent):
            if not c or not c.strip():
                continue
            neg = False
            piv = None
            m = _TV_NOTAS.search(c)
            if m:
                piv, neg = m, True
                span = (m.start(1), m.end(1))
            else:
                for rx in _TV_PIVOT:
                    piv = rx.search(c)
                    if piv:
                        break
                if piv is None and _TV_PREF_TO.search(c):
                    piv = list(re.finditer(r"\bto\b", c))[-1]
                if piv is None and prev_piv is not None:
                    piv = re.search(r"\b" + re.escape(prev_piv) + r"\b", c, re.I)
                if piv is None:
                    continue
                span = (piv.start(), piv.end())
                prev_piv = piv.group(0).lower()
            left, right = c[:span[0]], c[span[1]:]
            if m is None:
                neg = bool(_TV_NEG.search(left[-40:])) or bool(re.search(r"\bdislike|\bhate", c, re.I))
            for a1, a2 in _TV_ATTR.findall(c):
                a_ = (a1 or a2).lower()
                if a_ not in _TV_ATTR_SKIP:
                    attrs.add(_tv_attr(a_))
            rel.append((right, left) if neg else (left, right))
    if len(rel) < 1:
        return None, "unparsed"
    if len(attrs) > 1:
        return None, "ill_posed:different_attributes"
    # nodes: options by text; other entities by their distinctive content tokens
    sides = [s for r in rel for s in r[:2]]
    ent = [_tv_side_entity(s, opts_n) for s in sides]
    opt_tok = {t for on in opts_n for t in on.split()}
    black = set()
    for s, e in zip(sides, ent):
        if e is not None:
            black |= {_tv_sing(t) for t in _tv_norm(s).split()} - opt_tok
    keys = []
    for s, e in zip(sides, ent):
        if e is not None:
            keys.append(("opt", e))
        else:
            toks = {_tv_sing(t) for t in _tv_norm(s).split() if len(t) > 1} - black - _TV_STOP - opt_tok
            keys.append(("res", frozenset(toks)))
    # merge residual nodes sharing a token
    res_ids = {}
    groups: list[set] = []
    for k in keys:
        if k[0] == "res":
            if not k[1]:
                return None, "unparsed"
            hit = [g for g in groups if g & k[1]]
            merged = set(k[1]).union(*hit) if hit else set(k[1])
            groups = [g for g in groups if not (g & k[1])] + [merged]
    def node(k):
        if k[0] == "opt":
            return k
        for gi, g in enumerate(groups):
            if g & k[1]:
                return ("res", gi)
    edges = set()
    for j in range(0, len(keys), 2):
        w, l_ = node(keys[j]), node(keys[j + 1])
        if w == l_:
            return None, "unparsed"
        edges.add((w, l_))
    nodes = {x for e in edges for x in e}
    if any(("opt", i) not in nodes for i in range(len(opts))):
        if sum(("opt", i) in nodes for i in range(len(opts))) < 1:
            return None, "unparsed"
    reach = {n: {b for a, b in edges if a == n} for n in nodes}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            new = set().union(*(reach[m] for m in reach[n])) | reach[n]
            if new != reach[n]:
                reach[n], changed = new, True
    if any(n in reach[n] for n in nodes):
        return None, "ill_posed:cyclic"
    if any(("opt", i) not in nodes for i in range(len(opts))):
        return None, "unparsed:option_not_in_statements"
    if bottom:  # "who would you rather NOT invite?", "the worst player": the option every other option beats
        win = [i for i in range(len(opts)) if all(("opt", i) in reach[("opt", j)] for j in range(len(opts)) if j != i)]
    else:
        win = [i for i in range(len(opts)) if all(("opt", j) in reach[("opt", i)] for j in range(len(opts)) if j != i)]
    if len(win) == 1:
        return options.index(opts[win[0]]) if opts[win[0]] in options else [str(o).strip() for o in options].index(opts[win[0]]), "ok"
    return None, "ill_posed:undetermined"


def solve_transitivity(df: pd.DataFrame) -> pd.DataFrame:
    res = [solve_transitivity_one(q, [str(x) for x in o]) for q, o in zip(df.question_text, df.options)]
    return _out(df.index, [r[0] for r in res], [r[1] for r in res])


def fix_transitivity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    opts, corr, reps = [], [], []
    for o, c, r in zip(df.options, df.correct, df.repair):
        new, nlab, trunc = clean_transitivity_options(o)
        c = list(c)[:len(new)]
        r = list(r) + (["option_label_stripped"] if nlab else []) + (["options_truncated"] if trunc else [])
        opts.append(new); corr.append(c); reps.append(r)
    df["options"], df["correct"], df["repair"] = opts, corr, reps
    df["option_ids"] = df.options.map(lambda o: list(range(len(o))))
    return df


SOLVERS = {
    "transitivity": solve_transitivity,
    "sunk_cost": solve_sunk_cost,
    "completeness": solve_completeness,
    "completeness_risk": solve_completeness_risk,
    "time_inconsistency": solve_time_inconsistency,
    "transitivity_risk": solve_transitivity_risk,
}

_AX_POL = {"grading": "answer", "authoritative": True, "drop_status": ["ill_posed"], "shuffle": None}
POLICY = {el: dict(_AX_POL) for el in
          ["completeness", "completeness_risk", "transitivity", "transitivity_risk", "time_inconsistency", "sunk_cost"]}

FIXES = {"transitivity": fix_transitivity}

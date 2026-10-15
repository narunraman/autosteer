"""Solvers and repair hooks for the repeated-game / sequential-game elements (multi_agent.ipynb).

Elements: trigger, enforceability, feasibility, backward_induction.

Generator (release-era code = template_configs/multi_agent_utils.py @ 76b07f0 / ffa1840, functionally identical to
src/filler_tools/multi_agent_tools.py in the current tree): LLM-written templates filled with random payoffs.

Symmetric 2x2 stage games are written in words.  `parse_game` reads them into u(own, other) for the two actions
C ("cooperative": honor, collude, keep prices high, share, secure, ...) and D ("competitive": defect, breach,
price war, gamble, ...):  R = u(C,C), S = u(C,D), T = u(D,C), P = u(D,D).  Signs follow the text ("incurs a loss
of $X", "-$X", "X in costs/damages" -> negative).  It is a heuristic reader of LLM prose (action lexicon +
clause structure); rows it cannot read unambiguously get status "unparsed".

Numeric matching tolerance: trigger 0.006 on the 2-decimal discount factor; money values 0.01 relative 1e-9.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

REPAIRS = {
    "option_rounded": "option value printed with more than 2 decimals (answer leak) rounded to 2 decimals",
    "option_formatted": "option value printed with fewer decimals than the distractors (answer leak) padded to 2 decimals",
}

# --------------------------------------------------------------------------------------------- lexicon
C_WORDS = [
    r"cooperat\w*", r"collaborat\w*", r"collud\w*", r"collusi\w*", r"high[- ]pric\w*", r"high prices?",
    r"prices? high", r"maintain\w*", r"honou?r\w*", r"respect\w*", r"adher\w*", r"compl(?:y|ies|ied|ying|iant|iance)",
    r"uphold\w*", r"upheld", r"shar(?:e|es|ed|ing)", r"secur(?:e|es|ed|ing)", r"safe\w*", r"pocket\w*",
    r"lock(?:s|ed|ing)? in", r"cautious\w*", r"sustainab\w*", r"invest\w*", r"positive\w*", r"gradual\w*",
    r"trust\w*", r"conserv\w*", r"protect\w*(?! (?:its|their|own)\b)", r"guarantee\w*", r"stick\w*", r"stuck", r"contribut\w*",
    r"disarm\w*", r"peace\w*", r"fair\w*", r"passive\w*", r"limit\w*", r"restrain\w*", r"moderat\w*",
    r"keep(?:s|ing)? (?:the |their )?prices? high", r"clean\w*", r"reduc\w*", r"keep(?:s|ing)? (?:the |their )?(?:agreement|promise|pact|deal|treaty|contract)",
    r"loyal\w*", r"fixed", r"conservative\w*", r"open\w*", r"transparen\w*", r"disclos\w*", r"coordinat\w*",
    r"support\w*", r"friendl\w*", r"complementar\w*", r"regulat\w*", r"joint\w*", r"quota", r"green\w*", r"safeguard\w*", r"abid\w*",
    r"settl\w*", r"obey\w*", r"observ\w*", r"follow\w* (?:the )?(?:agreement|treaty|pact|guidelines?|rules?)",
]
D_WORDS = [
    r"defect\w*", r"compet\w*", r"price wars?", r"undercut\w*", r"low[- ]pric\w*", r"low(?:er|ers|ering)? (?:its |their |the )?prices?",
    r"lower prices?", r"breach\w*", r"reneg\w*", r"violat\w*", r"default\w*", r"tariff\w*", r"impos\w*",
    r"gambl\w*", r"risk\w*", r"bet(?:s|ting)?", r"roll\w*", r"frantic\w*", r"ignor\w*", r"break\w*", r"broke\w*",
    r"withdr[ae]w\w*", r"aggress\w*", r"push\w*", r"secret\w*", r"withh[oe]ld\w*", r"private\w*", r"espionage",
    r"pollut\w*", r"over-?fish\w*", r"cheat\w*", r"deviat\w*", r"exploit\w*", r"independen\w*", r"spy\w*", r"steal\w*",
    r"dump\w*", r"undermin\w*", r"attack\w*", r"arm(?:s|ing|ed)?\b", r"escalat\w*", r"hoard\w*", r"betray\w*",
    r"cut(?:s|ting)? corners?", r"aggressor", r"unilateral\w*", r"slash\w*", r"discount(?:ed|ing)? (?:its |their )?prices?",
    r"sabotag\w*", r"exclusive\w*", r"retaliat\w*", r"overproduc\w*", r"overexploit\w*", r"flood\w*", r"negative\w*", r"criticiz\w*",
    r"selfish\w*", r"free[- ]rid\w*", r"shirk\w*", r"exceed\w*", r"dishonou?r\w*", r"non-?compl\w*", r"noncompl\w*",
    r"relax\w*", r"reinvest\w*", r"protect\w* (?:its|their) own", r"disregard\w*", r"victim\w*", r"monopol\w*", r"unrestrict\w*", r"overus\w*", r"over-?\w+ing", r"renounc\w*",
]
_C = re.compile(r"\b(?:" + "|".join(C_WORDS) + r")", re.I)
_D = re.compile(r"\b(?:" + "|".join(D_WORDS) + r")", re.I)
_NEG = re.compile(r"\b(?:not|n't|never|no|neither|nor|fails? to|stop\w*|refus\w*|without|abandon\w*|instead of|refrain\w* from|abstain\w* from|avoid\w*|forgo\w*|cease\w*)\W+(?:\w+\W+){0,2}$", re.I)
_DISC = re.compile(r"discount|rate of|δ|delta|factor", re.I)
NUM = re.compile(r"(?<![\w.])(-\s?)?\$?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\w])")
_LOSS_PRE = re.compile(r"\b(?:los[se]\w*|lost|cost\w*|damage\w*|penalt\w*|suffer\w*|deficit\w*|incur\w*|pay(?:s|ing)?|fine[sd]?|"
                       r"debt\w*|negative|expens\w*)\W+(?:\w+\W+){0,4}$", re.I)
_LOSS_POST = re.compile(r"^\s*(?:dollars\s+|points\s+|units\s+|million\s+)?(?:in|of|as|worth of)\s+(?:\w+\s+){0,3}?"
                        r"(?:loss\w*|cost\w*|damage\w*|penalt\w*|expens\w*|fines?|debt\w*)", re.I)
_GAIN_PRE = re.compile(r"\b(?:sav\w*|gain\w*|earn\w*|benefit\w*|profit\w*|receiv\w*|get\w*|obtain\w*|reduc\w*)\W+(?:\w+\W+){0,1}$", re.I)


def _cls_hits(s: str):
    """List of (pos, 'C'|'D') action mentions in s; a negated cooperative word counts as D."""
    hits = []
    for m in _C.finditer(s):
        neg = _NEG.search(s[:m.start()][-40:]) is not None
        hits.append((m.start(), "D" if neg else "C"))
    for m in _D.finditer(s):
        # "fails to ... defect"/"not defect" -> C
        neg = _NEG.search(s[:m.start()][-40:]) is not None
        hits.append((m.start(), "C" if neg else "D"))
    return sorted(hits)


def action_class(s: str):
    h = [c for _, c in _cls_hits(s)]
    if not h:
        return None
    nc, nd = h.count("C"), h.count("D")
    if nc and not nd:
        return "C"
    if nd and not nc:
        return "D"
    return "mixed"


def first_class(s: str):
    h = _cls_hits(s)
    return h[0][1] if h else None


def payoff_numbers(text: str, drop_discount=True):
    """[(value, start, end)] of numbers in the text; discount factors (0<v<1 near 'discount') dropped;
    value negated if the text marks it as a loss/cost."""
    out = []
    for m in NUM.finditer(text):
        v = float(m.group(2).replace(",", ""))
        pre = text[max(0, m.start() - 60):m.start()]
        if drop_discount and 0 <= v < 1 and "." in m.group(2):
            continue
        if drop_discount and v < 1.0001 and _DISC.search(pre[-30:]):
            continue
        post = text[m.end():m.end() + 60]
        # ordinal/structural numbers ("Firm 1", "3 rounds") are not payoffs
        if re.match(r"\s*(?:rounds?|periods?|years?|players?|firms?|companies|countries|times|%)", post, re.I) and "$" not in m.group(0):
            continue
        neg = bool(m.group(1))
        if not neg:
            p40 = pre[-45:]
            if _LOSS_PRE.search(p40) and not re.search(r"\bsav\w*|\breduc\w*|\bavoid\w*", p40, re.I):
                neg = True
            elif _LOSS_POST.match(post) and not _GAIN_PRE.search(pre[-25:] if False else ""):
                if not re.search(r"\bsav\w*|\breduc\w*|\bavoid\w*|\bcut\w*", pre[-45:] + post[:30], re.I):
                    neg = True
        out.append((-v if neg else v, m.start(), m.end()))
    return out


_SPLIT = re.compile(r"(?<=[.;?!])\s+|;\s*|\s+(?=\b(?:[Ii]f|[Ss]hould|[Ww]hen|[Ww]hereas|[Oo]therwise)\b)")
_BOTH = re.compile(r"\b(?:both|each|neither|mutual\w*|all|two)\b", re.I)
_ONE = re.compile(r"\b(?:one|other|another|unilateral\w*|latter|former|while|whereas|vice versa|the [a-z]+(?:er|or)\b|firm [A-Z]\b|company [A-Z]\b|country [A-Z]\b)", re.I)
_RESP = re.compile(r"respective", re.I)


def _segments(text: str):
    """Split into clause segments; returns list of (start, end)."""
    cuts = [0] + [m.end() for m in _SPLIT.finditer(text)] + [len(text)]
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a:
            segs.append((a, b))
    return segs


def _tail(s: str):
    parts = re.split(r",|;|:|\bwhile\b|\bwhereas\b|\bresulting in\b|\band\b|\bbut\b|\bthen\b", s)
    return parts[-1]


_REL = re.compile(r"\b(?:that|who|which|whichever)\b(.*)$", re.I | re.S)


_ONEW = re.compile(r"\b(?:one|other|another|unilateral\w*|latter|former|respective\w*|vice versa)\b", re.I)
_POST_FOR = re.compile(r"^[^,;.\d]{0,25}?\b(?:for|to)\s+(?:the|a|an)\s+([^,;.\d]{0,60})", re.I)
_STOP_ENT = {"If", "The", "Then", "In", "When", "Should", "While", "Each", "Both", "They", "This", "It", "However",
             "Similarly", "Otherwise", "On", "Under", "Given", "What", "Which", "Assuming", "Consider", "Two"}


def _post_owner(post: str):
    """'# for the risk-taker' -> class of the phrase after 'for the'."""
    m = _POST_FOR.match(post)
    if not m:
        return None
    c = action_class(m.group(1))
    return c if c in ("C", "D") else None


def _entity_owner(tail: str, cond: str):
    ent = [e for e in re.findall(r"\b([A-Z][A-Za-z]*)\b", tail) if e not in _STOP_ENT]
    ent = [e for e in ent if e not in ("Firm", "Company", "Country", "Player", "Nation")]
    if not ent:
        return None
    e = ent[-1]
    for part in re.split(r",|\bwhile\b|\bwhereas\b|\band\b|\bbut\b", cond):
        mm = re.search(r"\b" + re.escape(e) + r"\b", part)
        if mm:
            c = first_class(part[mm.end():])
            if c:
                return c
    return None


def _owner(local: str, cond: str):
    """Action class of the player who receives the number, from the text just before it."""
    t = _tail(local)
    if re.search(r"\b(?:former|latter)\b", t, re.I):
        m = list(re.finditer(r"\b(?:one|if|should)\b", cond, re.I))
        h = _cls_hits(cond[m[-1].start():] if m else cond)
        if h:
            c = h[0][1]
            return c if re.search(r"\bformer\b", t, re.I) else ("D" if c == "C" else "C")
    m = _REL.search(t) or _REL.search(local)
    if m:  # "the country that over-fishes (while the other ...) is"
        c = first_class(m.group(1))
        if c:
            return c
    c = action_class(t)
    if c in ("C", "D"):
        return c
    return _entity_owner(t, cond)


_MROW = re.compile(r"^\s*(Action \w+)\s+\(\s*(-?[\d.]+),\s*(-?[\d.]+)\)\s+\(\s*(-?[\d.]+),\s*(-?[\d.]+)\)\s*$", re.M)


def parse_matrix(text: str):
    """2x2 bimatrix printed by pandas: rows 'Action X  (a, b)  (c, d)'.  Returns (labels, u) with
    u[i][j] = (row payoff, column payoff), or None."""
    rows = _MROW.findall(text)
    if len(rows) != 2:
        return None
    labels = [r[0] for r in rows]
    u = [[(float(r[1]), float(r[2])), (float(r[3]), float(r[4]))] for r in rows]
    return labels, u


def _matrix_game(text: str):
    pm = parse_matrix(text)
    if pm is None:
        return None, "missing_payoffs"
    labels, u = pm
    if not (u[0][1][0] == u[1][0][1] and u[0][1][1] == u[1][0][0] and u[0][0][0] == u[0][0][1] and u[1][1][0] == u[1][1][1]):
        return None, "not_symmetric"
    g0 = {"R": u[0][0][0], "S": u[0][1][0], "T": u[1][0][0], "P": u[1][1][0]}   # action 0 cooperative
    g1 = {"R": u[1][1][0], "S": u[1][0][0], "T": u[0][1][0], "P": u[0][0][0]}   # action 1 cooperative
    # which action is the cooperative one: named in the strategy ("play Action X ... until ... Action Y")
    m = re.search(r"(Action \w+)[^.?]*?\b(?:until|unless|so long as|as long as|until either|after which|otherwise)\b", text)
    if m and m.group(1) in labels:
        return (g0 if m.group(1) == labels[0] else g1), None
    # not named ("grim trigger strategies"): the cooperative action is the one that makes the game a PD
    pd0, pd1 = [g["T"] > g["R"] > g["P"] > g["S"] for g in (g0, g1)]
    if pd0 != pd1:
        return (g0 if pd0 else g1), None
    return None, "cooperative_action_unnamed"


def _order_owner(cond: str):
    """'if one X while the other Y, they get a and b respectively' -> class of X (owner of a)."""
    m = list(re.finditer(r"\b(?:one|either|if|should|when)\b", cond, re.I))
    cm = cond[m[-1].start():] if m else cond
    for w in ("one", "if", "should", "when"):  # prefer the 'one ...' actor phrase
        mm = list(re.finditer(r"\b" + w + r"\b", cond, re.I))
        if mm:
            cm = cond[mm[-1].start():]
            break
    parts = re.split(r"\bwhile\b|\band\b|,|\bwhereas\b|\bbut\b|\bas\b", cm)
    h = _cls_hits(parts[0])
    if h:  # "invests in aggressive R&D" -> the competitive modifier wins
        return "D" if "D" in {c for _, c in h} else "C"
    if len(parts) > 1:
        h = _cls_hits(parts[1])
        if h:
            return "D" if h[-1][1] == "C" else "C"
    return None


def parse_game(text: str):
    """Read a symmetric 2x2 game from prose. Return (dict(R,S,T,P), None) or (None, reason)."""
    if re.search(r"^\s*Action \w+\s+\(", text, re.M) or re.search(r"payoff matrix|stage game|one-shot payoff matrix|matrix", text, re.I) \
            and len(payoff_numbers(text)) < 4:
        return _matrix_game(text)
    nums = payoff_numbers(text)
    if len(nums) < 4:
        return None, "missing_payoffs"
    items = []
    for a, b in _segments(text):
        ins = [n for n in nums if a <= n[1] < b]
        prev = a
        for k, (v, s0, e0) in enumerate(ins):
            nxt = ins[k + 1][1] if k + 1 < len(ins) else b
            items.append(dict(v=v, s=s0, e=e0, a=a, k=k, local=text[prev:s0], pre=text[a:s0], post=text[e0:nxt]))
            prev = e0
    diag, off = [], []
    for it in items:
        ctx = it["pre"] if it["k"] == 0 else it["local"]
        ctx = re.sub(r"^\W*respectively\b", "", ctx)
        post_each = re.match(r"\s*(?:in \w+ )?(?:to )?(?:each|apiece)\b", it["post"]) is not None
        both = _BOTH.search(ctx) is not None or post_each
        if both and not _ONEW.search(ctx):
            c = action_class(ctx)
            if c is None and post_each:
                c = action_class(it["post"])
            diag.append((it["v"], c))
        else:
            off.append(it)
    if len(off) == 4 and {off[0]["v"], off[1]["v"]} == {off[2]["v"], off[3]["v"]}:
        off = off[:2]          # "A ... while B ..., x, y; B ... while A ..., y, x"
    if len(diag) != 2 or len(off) != 2:
        return None, f"structure:{len(diag)}d{len(off)}o"
    ix, iy = off
    cond = text[ix["a"]:ix["s"]] if ix["a"] == iy["a"] else ix["pre"]
    resp = _RESP.search(ix["pre"][-60:] + ix["post"] + iy["post"][:40]) is not None
    ox, oy = _post_owner(ix["post"]), _post_owner(iy["post"])
    if not (ox or oy):
        if resp:
            ox = _order_owner(cond)
        else:
            ox = _owner(ix["local"] if ix["k"] else ix["pre"], cond)
            oy = _owner(iy["local"] if iy["k"] else iy["pre"], cond if ix["a"] == iy["a"] else iy["pre"])
            if ox and oy and ox == oy:
                return None, "offdiag_roles"
    if ox is None and oy in ("C", "D"):
        ox = "D" if oy == "C" else "C"
    if oy is None and ox in ("C", "D"):
        oy = "D" if ox == "C" else "C"
    if {ox, oy} != {"C", "D"}:
        return None, "offdiag_roles"
    (d1, c1), (d2, c2) = diag
    if c1 not in ("C", "D") and c2 in ("C", "D"):
        c1 = "D" if c2 == "C" else "C"
    if c2 not in ("C", "D") and c1 in ("C", "D"):
        c2 = "D" if c1 == "C" else "C"
    if {c1, c2} != {"C", "D"}:
        return None, "diag_roles"
    R, P = (d1, d2) if c1 == "C" else (d2, d1)
    T, S = (ix["v"], iy["v"]) if ox == "D" else (iy["v"], ix["v"])
    return {"R": R, "S": S, "T": T, "P": P}, None


def _out(index, preds, stats):
    return pd.DataFrame({"pred": pd.array(preds, dtype="Int64"), "status": stats}, index=index)


def _num(s: str):
    m = re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(s))
    return [float(x.replace(",", "")) for x in m]


# --------------------------------------------------------------------------------------------- trigger
def trigger_delta(g):
    """Minimum grim-trigger discount factor, or (None, reason) when grim trigger is not an SPNE/ill-posed."""
    R, S, T, P = g["R"], g["S"], g["T"], g["P"]
    if not (R > P):
        return None, "not_pd:mutual_defection_not_worse"      # (D,D) >= (C,C): nothing to sustain
    if P < S:
        return None, "not_pd:punishment_not_nash"              # (D,D) not a stage NE -> grim trigger never SPNE
    if T <= R:
        return None, "not_pd:no_temptation"                    # cooperation is a stage NE, delta* = 0
    return (T - R) / (T - P), None


def solve_trigger(df):
    preds, stats = [], []
    for text, opts in zip(df.question_text, df.options):
        g, why = parse_game(text)
        if g is None:
            preds.append(pd.NA)
            stats.append("ill_posed:missing_payoffs" if why == "missing_payoffs" else f"unparsed:{why}")
            continue
        d, why = trigger_delta(g)
        if d is None:
            preds.append(pd.NA)
            stats.append(f"ill_posed:{why}")
            continue
        vals = [(_num(o) or [np.nan])[0] for o in opts]
        hit = [i for i, v in enumerate(vals) if abs(v - round(d, 2)) < 0.006]
        if not hit:
            preds.append(pd.NA)
            stats.append("no_match")
        elif len({opts[i] for i in hit}) > 1:
            preds.append(pd.NA)
            stats.append("multi_match")
        else:
            preds.append(hit[0])
            stats.append("ok")
    return _out(df.index, preds, stats)


def fix_trigger(df):
    """The correct option is f"{round(d, 2)}" (e.g. '0.5') while distractors are f"{x:.2f}" ('0.50'): pad to 2 decimals."""
    df = df.copy()
    new = [[f"{float(o):.2f}" if re.fullmatch(r"\d?\.\d|\d", o.strip()) else o for o in opts] for opts in df.options]
    ch = pd.Series([list(a) != list(b) for a, b in zip(new, df.options)], index=df.index)
    df["options"] = new
    df.loc[ch, "repair"] = df.loc[ch, "repair"].map(lambda r: r + ["option_formatted"])
    return df


SOLVERS = {"trigger": solve_trigger}
FIXES = {"trigger": fix_trigger}
# All four generators compute the stored answer from the tag / a formula that ignores the displayed game's
# structure (non-PD games, IR ignored, broken backward induction), so the stored answers are unreliable wherever
# the solver cannot read the game: authoritative solvers, drop everything that is not "ok".
_P = {"grading": "answer", "authoritative": True, "drop_status": ["ill_posed", "unparsed", "no_match", "multi_match"],
      "shuffle": None}
POLICY = {el: dict(_P) for el in ("trigger", "enforceability", "feasibility", "backward_induction")}


# --------------------------------------------------------------------------------------------- backward induction
_BI_NUM = re.compile(r"(?<![\w.])(-\s?)?\$?(\d+(?:\.\d+)?)(?![\w])")
_STRUCT_PRE = re.compile(r"(?:round|stage|phase|step|decision point|node|player|level|year|month|day|week|dr\.?)\s*\W?\s*$", re.I)
_STRUCT_POST = re.compile(r"\s*(?:\w+[- ])?(?:rounds?|stages?|phases?|steps?|decisions?|investors?|players?|parties|doctors?|%|"
                          r"percent|years?|months?|days?|weeks?|times)\b|(?:nd|rd|th|st)\b|-(?:round|stage)", re.I)
_END_W = re.compile(r"\b(?:finali[sz]\w*|final(?:ly)? (?:agree|accept)|conclud\w*|close\w*|clos(?:e|ing)\b|end\w*|accept\w*|agree\w*|"
                    r"settl\w*|seal\w*|secur\w*|sign\w*|lock\w*|stop\w*|take\w*|revert\w*|immediately|approv\w*|commit\w*|"
                    r"stick\w*|keep\w*|maintain\w*|implement\w*|endorse\w*|confirm\w*|complet\w*)\b", re.I)
_CONT_W = re.compile(r"\b(?:continu\w*|pass\w*|push\w*|defer\w*|hand\w*|let\w*|wait\w*|extend\w*|further|reject\w*|declin\w*|"
                     r"refus\w*|delay\w*|return\w*|back to|allow\w*|escalat\w*|renegotiat\w*|persist\w*|propos\w*|refer\w*|"
                     r"postpon\w*|hold\w* out|another round|next round|third round|final round|more)\b", re.I)


def bi_numbers(text: str):
    out = []
    for m in _BI_NUM.finditer(text):
        pre, post = text[max(0, m.start() - 25):m.start()], text[m.end():m.end() + 30]
        if _STRUCT_PRE.search(pre) or _STRUCT_POST.match(post):
            continue
        v = float(m.group(2))
        neg = bool(m.group(1))
        if not neg and _LOSS_PRE.search(pre[-30:]) and not re.search(r"\bsav\w*|\breduc\w*", pre[-30:], re.I):
            neg = True
        out.append((-v if neg else v, m.start(), m.end()))
    return out


def _players(text: str, opts):
    """(first mover, second mover) names."""
    names = re.findall(r"\b(?:Dr\.|Player|Physician|Doctor|Investor|Partner|Firm|Company)\s[A-Z][A-Za-z]*\b|\b[A-Z][A-Za-z]+\b", text)
    names = [n for n in names if n not in ("Dr", "Player", "Physician", "Doctor", "Investor", "Partner", "Firm", "Company")]
    stop = {"In", "If", "The", "Two", "Should", "However", "Both", "Using", "Given", "What", "Considering", "Consider",
            "Alternatively", "Otherwise", "During", "At", "After", "Each", "This", "They", "She", "He", "Her", "His",
            "Initially", "Finally", "Assuming", "Imagine", "Once", "When", "With", "Then", "For", "Based", "Faced",
            "Utilizing", "Round", "Stage", "As", "On", "An", "A", "Under", "Their", "It", "Suppose", "From", "Now",
            "Following", "Being", "Before", "To", "Instead", "Similarly", "Here", "There", "Meanwhile", "Also", "Such"}
    names = [n for n in names if n.split()[-1] not in stop and n not in stop]
    full = {}
    for n in names:
        k = n.split()[-1]
        if k not in full or len(n) > len(full[k]):
            full[k] = n
    names = [full[n.split()[-1]] for n in names]
    cnt = {}
    for n in names:
        cnt[n] = cnt.get(n, 0) + 1
    top = sorted(cnt, key=lambda n: (-cnt[n], text.find(n)))[:2]
    if len(top) == 2 and cnt[top[1]] < 3:
        return []
    return sorted(top, key=lambda n: text.find(n))   # first mover is introduced first


def _pair_owner_swapped(text, nx, ny, p1, p2):
    """True if the pair (x, y) as written gives p2's payoff first."""
    lx = text[max(0, nx[1] - 120):nx[1]]
    ly = text[nx[2]:ny[1]]
    post_x = text[nx[2]:nx[2] + 40]
    post_y = text[ny[2]:ny[2] + 40]

    def last_name(s):
        best, pos = None, -1
        for n, lab in ((p1, 1), (p2, 2)):
            for m in re.finditer(re.escape(n), s):
                if m.start() > pos:
                    best, pos = lab, m.start()
        return best
    # "X for Bob" / "X for herself"
    FOR = r"\s*(?:\w+\s){0,3}?for\s+(?:(?:her|his|their)\s+(?:firm|company)|([A-Z][\w.]*(?:\s[A-Z]\w*)?))"
    fx, fy = re.match(FOR, post_x), re.match(FOR, post_y)
    if fx and fx.group(1):
        if p2.split()[-1] in fx.group(1):
            return True
        if p1.split()[-1] in fx.group(1):
            return False
    if fy and fy.group(1):
        if p1.split()[-1] in fy.group(1):
            return True
        if p2.split()[-1] in fy.group(1):
            return False
    if re.search(r"respectively", post_y) or re.search(r"respectively", ly):
        s = lx[-120:]
        i1, i2 = s.rfind(p1), s.rfind(p2)
        # "Alice and Bob would receive x and y respectively"
        m = re.search(re.escape(p2) + r"\s+and\s+" + re.escape(p1), s)
        return bool(m)
    ox = last_name(_tail(lx[-90:]))
    oy = last_name(ly)
    if oy is not None:
        return oy == 1
    if ox is not None:
        return ox == 2
    return False


def parse_bi(text: str, opts):
    if re.search(r"\\?\(p\d_round|\{p\d_round", text):
        return None, "ill_posed:unfilled_template"
    nums = bi_numbers(text)
    if len(nums) < 8:
        return None, "ill_posed:missing_payoffs"
    if len(nums) > 8:
        return None, "unparsed:extra_numbers"
    pl = _players(text, opts)
    if len(pl) < 2:
        return None, "unparsed:players"
    p1, p2 = pl[0], pl[1]
    pairs = []
    # round-3 written as "Alice gets either a or a', Bob either b or b'"
    if re.search(r"\beither\b", text[nums[3][2]:nums[4][1]]) and re.search(r"\beither\b", text[nums[5][2]:nums[6][1]]):
        a, a2, b, b2 = [n[0] for n in nums[4:8]]
        raw = [(nums[0], nums[1]), (nums[2], nums[3])]
        tail = [(a, b), (a2, b2)]
    else:
        raw = [(nums[i], nums[i + 1]) for i in (0, 2, 4, 6)]
        tail = None
    for nx, ny in raw:
        sw = _pair_owner_swapped(text, nx, ny, p1, p2)
        pairs.append((ny[0], nx[0]) if sw else (nx[0], ny[0]))
    if tail:
        pairs += tail
    return {"p1": p1, "p2": p2, "pairs": pairs, "_nums": nums}, None


def bi_solve_game(pairs):
    """pairs = [r1_end, r2_end, r3_a, r3_b] as (u1, u2). Returns (choice per round {1,2,3}, reason)."""
    r1, r2, r3a, r3b = pairs
    if r3a[0] == r3b[0] and r3a[1] != r3b[1]:
        return None, "tie_round3"
    v3 = r3a if r3a[0] >= r3b[0] else r3b
    if r2[1] == v3[1]:
        return None, "tie_round2"
    ch2 = "end" if r2[1] > v3[1] else "continue"
    v2 = r2 if ch2 == "end" else v3
    if r1[0] == v2[0]:
        return None, "tie_round1"
    ch1 = "end" if r1[0] > v2[0] else "continue"
    return {1: ch1, 2: ch2}, None


def _asked_round(text: str, tags):
    q = re.split(r"(?<=[.!])\s+", text.strip())[-1]
    if re.search(r"\b(?:first|1st|initial\w*|opening|very first|round (?:1|one)|start\w*|outset|begin\w*)\b", q, re.I):
        return 1
    if re.search(r"\b(?:second|2nd|round (?:2|two)|next round|middle)\b", q, re.I):
        return 2
    if re.search(r"\b(?:third|final|last|round (?:3|three))\b", q, re.I):
        return 3
    t = ";".join(tags or [])
    return 1 if "round1" in t else 2 if "round2" in t else None


def _opt_player(o: str, p1: str, p2: str):
    head = o.split("should", 1)[0]
    if p2 in head or p2.split()[-1] in head.split():
        return 2
    if p1 in head or p1.split()[-1] in head.split():
        return 1
    return None


def _opt_kind(o: str):
    body = o.split("should", 1)[-1]
    e, c = bool(_END_W.search(body)), bool(_CONT_W.search(body))
    if e and not c:
        return "end"
    if c and not e:
        return "continue"
    if e and c:
        # "accept ... and let Bob ..." : first verb wins
        me, mc = _END_W.search(body).start(), _CONT_W.search(body).start()
        return "end" if me < mc else "continue"
    return None


_STOPW = {"should", "their", "with", "the", "and", "for", "that", "this", "from", "into", "will", "would", "have", "they",
          "them", "then", "back", "own", "his", "her"}


def _words(s: str, drop=()):
    w = {x[:5] for x in re.findall(r"[a-z]{4,}", s.lower()) if x not in _STOPW}
    return w - {x[:5] for d in drop for x in re.findall(r"[a-z]{4,}", d.lower())}


def _cond_clause(text: str, pos: int, after: bool):
    """Conditional clause ('If X does Y,') just before pos, or the first one after pos."""
    text = text.replace("Dr.", "Dr_").replace("Mr.", "Mr_").replace("Ms.", "Ms_").replace("Corp.", "Corp_")
    if after:
        m = re.search(r"\b(?:If|if|Should|should|When|when|Once|once)\b([^,.;]*)", text[pos:])
        return m.group(1) if m else ""
    ms = list(re.finditer(r"\b(?:If|if|Should|should|When|when|Once|once|[Ii]n case)\b([^,.;]*)", text[:pos]))
    return ms[-1].group(1) if ms else ""


def _overlap_kinds(text, opts, g, rnd):
    """Which option ends the game at round rnd: the one sharing words with the condition that introduces the
    round-rnd end payoffs ('If Dr. A keeps the usual treatment, ...') rather than the continuation condition."""
    nums = g["_nums"]
    if rnd not in (1, 2):
        return None
    i_end = nums[2 * (rnd - 1)]
    i_next = nums[2 * (rnd - 1) + 1]
    end_c = _cond_clause(text, i_end[1], after=False)
    cont_c = _cond_clause(text, i_next[2], after=True)
    names = [g["p1"], g["p2"]]
    sc = []
    for o in opts:
        w = _words(o.split("should", 1)[-1], drop=names)
        sc.append(len(w & _words(end_c, names)) - len(w & _words(cont_c, names)))
    if len(opts) != 2 or sc[0] == sc[1]:
        return None
    return ["end", "continue"] if sc[0] > sc[1] else ["continue", "end"]


def solve_backward_induction(df):
    preds, stats = [], []
    for text, opts, tags, dom in zip(df.question_text, df.options, df.tags, df.domain):
        g, why = parse_bi(text, opts)
        if g is not None:
            g["domain"] = dom
        if g is None:
            preds.append(pd.NA)
            stats.append(why)
            continue
        ch, why = bi_solve_game(g["pairs"])
        if ch is None:
            preds.append(pd.NA)
            stats.append(f"ill_posed:{why}")
            continue
        rnd = _asked_round(text, list(tags) if tags is not None else [])
        who = {_opt_player(o, g["p1"], g["p2"]) for o in opts} - {None}
        if who == {2}:
            rnd = 2
        elif who == {1} and rnd == 2:
            rnd = None
        if rnd not in (1, 2):
            preds.append(pd.NA)
            stats.append("unparsed:round" if rnd is None else "ill_posed:asks_round3")
            continue
        kinds = [_opt_kind(o) for o in opts]
        ov = _overlap_kinds(text, opts, g, rnd)
        if str(g.get("domain", "")) == "medical" or sorted(k for k in kinds if k) != ["continue", "end"]:
            kinds = ov
        if kinds is None or sorted(k for k in kinds if k) != ["continue", "end"]:
            preds.append(pd.NA)
            stats.append("unparsed:options")
            continue
        preds.append(kinds.index(ch[rnd]))
        stats.append("ok")
    return _out(df.index, preds, stats)


SOLVERS["backward_induction"] = solve_backward_induction


# --------------------------------------------------------------------------------------------- repeated-game helpers
_DELTA = re.compile(r"(?<![\d.])(0?\.\d+)(?![\d])")


def discount(text: str):
    """The stated discount factor (the unique number in (0,1) in the text), or None."""
    v = {float(x) for x in _DELTA.findall(text) if 0 < float(x) < 1}
    return v.pop() if len(v) == 1 else None


def minmax(g):
    """Pure-strategy minmax of the symmetric stage game (the opponent picks the action minimising my best reply)."""
    return min(max(g["R"], g["T"]), max(g["S"], g["P"]))


def is_pd(g):
    return g["T"] > g["R"] > g["P"] > g["S"]


# --------------------------------------------------------------------------------------------- enforceability
_ALT = re.compile(r"\balternat\w*|\btakes? turns?\b|\bin turns?\b|\bturn[- ]taking\b|\bswitch\w*|\brotat\w*", re.I)
_ASYM = re.compile(r"\bone (?:firm|company|country|nation|player|party|side|of)\b|\bthe other\b|\bfirm [AB]\b|\bcompany [AB]\b|"
                   r"\bcountry [AB]\b|\bplayer [AB12]\b|\bwhile\b|\bwhereas\b", re.I)
_BOTHO = re.compile(r"\bboth\b|\beach\b|\ball\b|\bmutual\w*|\bneither\b|\bthey\b|\bjointly\b", re.I)


_OPT_D = re.compile(r"\b(?:price|fare|pricing|trade|bidding|arms) wars?\b|\bunder-?deliver\w*|\bunder-?invest\w*|\bdefect\w*|"
                    r"\bbreach\w*|\bcompet\w*|\bundercut\w*|\blow(?:er)? (?:prices?|fares?)\b", re.I)


def enf_option_kind(o: str):
    """'CC' | 'DD' | 'asym' | 'alt' | None for an enforceability option text."""
    if _ALT.search(o):
        return "alt"
    if _ASYM.search(o):
        return "asym"
    if _BOTHO.search(o) or re.search(r"\b(?:firms|companies|countries|nations|players|parties)\b", o, re.I):
        if _OPT_D.search(o):
            return "DD"
        c = first_class(o)
        return {"C": "CC", "D": "DD"}.get(c)
    return None


def stage_ne_payoffs(g):
    """Own payoffs of every pure Nash equilibrium of the symmetric 2x2 stage game, per player: list of (u1, u2)."""
    R, S, T, P = g["R"], g["S"], g["T"], g["P"]
    ne = []
    if R >= T:
        ne.append((R, R))
    if P >= S:
        ne.append((P, P))
    if S >= P and T >= R:  # (C,D) and (D,C)
        ne += [(S, T), (T, S)]
    return ne


def enforceable(g, delta, tol=1e-9):
    """Which stationary / alternating paths are subgame-perfect outcomes of the delta-discounted repeated game
    with grim-trigger reversion to a stage-game Nash equilibrium. The punishment is the
    worst pure stage NE for the deviator (reversion to a NE is itself subgame perfect). One-shot deviation
    condition on normalised discounted values. Returns dict kind -> bool, or None if the stage game has no pure NE."""
    R, S, T, P = g["R"], g["S"], g["T"], g["P"]
    ne = stage_ne_payoffs(g)
    if not ne:
        return None
    m = min(u[0] for u in ne)  # symmetric game: worst NE payoff of a player (same for both players)
    d = delta

    def ok(value, dev):  # value on path >= (1-d)*best one-shot deviation + d*punishment
        return value >= (1 - d) * dev + d * m - tol
    out = {
        "CC": ok(R, max(R, T)),
        "DD": ok(P, max(P, S)),
        # the D-player earns T (deviation to C gives R), the C-player earns S (deviation to D gives P)
        "asym": ok(T, max(T, R)) and ok(S, max(S, P)),
    }
    # (C,D),(D,C),...: normalised value when about to receive S / T
    vS, vT = (S + d * T) / (1 + d), (T + d * S) / (1 + d)
    out["alt"] = ok(vS, max(S, P)) and ok(vT, max(T, R))
    return out


def solve_enforceability(df):
    preds, stats = [], []
    for text, opts in zip(df.question_text, df.options):
        g, why = parse_game(text)
        if g is None:
            preds.append(pd.NA)
            stats.append("ill_posed:missing_payoffs" if why == "missing_payoffs" else f"unparsed:{why}")
            continue
        delta = discount(text)
        if delta is None:
            preds.append(pd.NA)
            stats.append("unparsed:discount")
            continue
        kinds = [enf_option_kind(o) for o in opts]
        if sorted(k for k in kinds if k) != sorted(set(k for k in kinds if k)) or None in kinds:
            preds.append(pd.NA)
            stats.append("unparsed:options")
            continue
        e = enforceable(g, delta)
        if e is None:
            preds.append(pd.NA)
            stats.append("ill_posed:no_pure_stage_ne")
            continue
        if "alt" in kinds and e["alt"]:
            # "take turns" semantics still open: keep the row with its stored answer, unverified
            preds.append(pd.NA)
            stats.append("open:take_turns")
            continue
        hit = [i for i, k in enumerate(kinds) if e[k]]
        if len(hit) == 1:
            preds.append(hit[0])
            stats.append("ok")
        elif not hit:
            preds.append(pd.NA)
            stats.append("ill_posed:none_enforceable")
        else:
            preds.append(pd.NA)
            stats.append("ill_posed:multi_enforceable:" + "+".join(sorted(kinds[i] for i in hit)))
    return _out(df.index, preds, stats)


SOLVERS["enforceability"] = solve_enforceability


# --------------------------------------------------------------------------------------------- feasibility
def _clip(poly, a, b, c):
    """Sutherland-Hodgman: keep the part of convex polygon `poly` with a*x + b*y >= c."""
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def fir_set(g):
    """Feasible and individually rational payoff set: convex hull of the stage payoff profiles intersected with
    u_i >= pure minmax.  Returns the polygon (list of vertices, counter-clockwise), possibly degenerate."""
    R, S, T, P = g["R"], g["S"], g["T"], g["P"]
    m = minmax(g)
    poly = _hull([(R, R), (S, T), (T, S), (P, P)])
    poly = _clip(_clip(poly, 1, 0, m), 0, 1, m)
    return poly, m


def in_poly(poly, x, y, tol):
    if not poly:
        return False
    if len(poly) < 3:
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        if len(poly) == 1:
            return abs(x - xs[0]) <= tol and abs(y - ys[0]) <= tol
        (x0, y0), (x1, y1) = poly
        L2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
        t = min(1, max(0, ((x - x0) * (x1 - x0) + (y - y0) * (y1 - y0)) / L2))
        return np.hypot(x - x0 - t * (x1 - x0), y - y0 - t * (y1 - y0)) <= tol
    for i in range(len(poly)):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
        L = np.hypot(x1 - x0, y1 - y0)
        # skip degenerate edges: clipping at u_i >= minmax through a hull vertex leaves near-duplicate vertices
        # (e.g. (P, P) and (P, P - 4e-12)), whose ~0-length edge has an arbitrary direction
        if L > 1e-6 and ((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) / L < -tol:
            return False
    return True


_NEGQ = re.compile(r"\b(?:can(?:'|’)?t|cannot|can not|not|infeasible|impossible|never|unattainable|unachievable)\b", re.I)
_PAIR = re.compile(r"^\s*\(\s*(-?\$?-?[\d,]+(?:\.\d+)?)\s*,\s*(-?\$?-?[\d,]+(?:\.\d+)?)\s*\)\s*$")
_SINGLE = re.compile(r"^\s*(-?\$?-?[\d,]+(?:\.\d+)?)(?:\s+each)?\s*$")


def _money(s):
    s = s.replace("$", "").replace(",", "")
    neg = s.count("-") % 2 == 1
    return -float(s.replace("-", "")) if neg else float(s.replace("-", ""))


def feas_asks_infeasible(text: str):
    q = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())[-1]
    q = q if "?" in q or re.search(r"\bwhich\b", q, re.I) else text
    return _NEGQ.search(q) is not None


def solve_feasibility(df, tol=0.01):
    """Option = payoff profile '($x, $y)' (in the FIR set?) or single payoff '$x' (read as the symmetric profile
    ($x, $x)).  Tolerance 0.01 (options are given to the cent)."""
    preds, stats = [], []
    for text, opts in zip(df.question_text, df.options):
        g, why = parse_game(text)
        if g is None:
            preds.append(pd.NA)
            stats.append("ill_posed:missing_payoffs" if why == "missing_payoffs" else f"unparsed:{why}")
            continue
        poly, m = fir_set(g)
        xs = [p[0] for p in poly]
        vals = []
        for o in opts:
            mp, ms = _PAIR.match(o), _SINGLE.match(o)
            if mp:
                vals.append(in_poly(poly, _money(mp.group(1)), _money(mp.group(2)), tol))
            elif ms:  # single payoff = the symmetric profile ($x, $x)
                v = _money(ms.group(1))
                vals.append(bool(xs) and in_poly(poly, v, v, tol))
            else:
                vals.append(None)
        if None in vals:
            preds.append(pd.NA)
            stats.append("unparsed:options")
            continue
        want = not feas_asks_infeasible(text)
        hit = [i for i, v in enumerate(vals) if v == want]
        if len(hit) == 1:
            preds.append(hit[0])
            stats.append("ok")
        else:
            preds.append(pd.NA)
            stats.append(f"ill_posed:{'none' if not hit else 'multi'}_{'feasible' if want else 'infeasible'}")
    return _out(df.index, preds, stats)


SOLVERS["feasibility"] = solve_feasibility


# --------------------------------------------------------------------------------------------- feasibility fixes
_LONGDEC = re.compile(r"(\d+\.\d{3,})")


def _round_opt(o: str) -> str:
    return _LONGDEC.sub(lambda m: f"{float(m.group(1)):.2f}", o)


def feas_tags(text: str, opts) -> str:
    kind = "infeasible" if feas_asks_infeasible(text) else "feasible"
    prof = any(_PAIR.match(o) for o in opts)
    return f"explicit;{kind}" + (";profile" if prof else "")


def fix_feasibility(df):
    df = df.copy()
    # 1. metadata: the release has 9,970 metadata rows for 9,974 questions; only 212 join by qid and even those
    #    are misaligned (their tags contradict the text/options in 167/212) -> derive for every row
    derived = [feas_tags(t, o) for t, o in zip(df.question_text, df.options)]
    miss = pd.Series(True, index=df.index)
    df["tags_raw"] = derived
    df.loc[miss, "type"] = "symmetric"
    df.loc[miss, "domain"] = "finance"
    df.loc[miss, "difficulty_raw"] = "7"
    df.loc[miss, "repair"] = df.loc[miss, "repair"].map(lambda r: r + ["meta_derived"])
    # 2. unrounded option values (the 'feasible;profile' distractors add uniform(-1000,1000) noise) leak the answer
    new = [[_round_opt(o) for o in opts] for opts in df.options]
    ch = pd.Series([list(a) != list(b) for a, b in zip(new, df.options)], index=df.index)
    df["options"] = new
    df.loc[ch, "repair"] = df.loc[ch, "repair"].map(lambda r: r + ["option_rounded"])
    return df


FIXES["feasibility"] = fix_feasibility

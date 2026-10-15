"""Solvers and repair hooks for the arithmetic / probability elements (v1-era, no creation code).

Elements: add_sub, mult_div, compute_expectations, bayes_rule, complement_rule, prob_basic, gamblers_fallacy.

Findings in brief (numbers in steer/elements/<element>.yaml):
  add_sub / mult_div   stored answer = a fixed formula of the two slot values (add |a|+|b|, both |a|-|b|,
                       sub -|a|-|b|; multiply a*b, divide a/b) regardless of the wording, so word problems whose
                       natural answer is b-a, a, a+b, b/a, ... are wrong (~30% of 'both' and ~80% of 'sub' story
                       rows in a hand-judged sample). mult_div negative*negative stores -(ab) or a+b.
                       Only verifiable rows are kept: explicit arithmetic phrasing (solver) and the two story
                       families validated by hand (add_sub 'add', mult_div 'multiply'+'positive').
  compute_expectations correct option lacks the unit and has fewer decimals than the distractors (answer leak).
  bayes_rule           domain labels cycle meaninglessly; 37% of generated number triples are incoherent.
  complement_rule      17 of 188 templates are ill-posed (non-disjoint events, garbled numbers, underdetermined).
  prob_basic           hand-written: every row judged by hand (tables below).
  gamblers_fallacy     every answer is "No"; 38 of 208 templates are unsound (without-replacement card draws,
                       non-random opponents, non-yes/no questions, literal reading gives "Yes").
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

REPAIRS = {
    "option_format": "option texts normalised to one format (same unit and number of decimals for all options)",
    "option_rounded": "unrounded correct option rounded to 2 decimals like the distractors",
    "answer_sign_fixed": "stored product of two negatives had the wrong sign; all options negated (distractors kept)",
    "answer_manual": "stored answer judged wrong by hand (prob_basic); corrected to the right option",
    "option_marker_removed": "option text contained '(Correct Option)'; marker removed and used as the answer",
    "option_garbage_removed": "option list contained fragments of another question; fragments removed",
}


# =============================================================================================== helpers
def _out(index, preds, stats):
    return pd.DataFrame({"pred": pd.array(preds, dtype="Int64"), "status": stats}, index=index)


def _add_repair(df, mask, rid):
    mask = np.asarray(mask, bool)
    if mask.any():
        df.loc[mask, "repair"] = df.loc[mask, "repair"].map(lambda r: list(r) + [rid])


_UNIT = re.compile(r"^\s*(-?\$?-?\d[\d,]*(?:\.\d+)?)\s*(%|points|days|times)?\s*$")


def opt_val(s) -> float:
    """Numeric value of an option like '-15.5', '8.2 days', '59.2%', '$3', else NaN."""
    m = _UNIT.match(str(s))
    if not m:
        return np.nan
    x = m.group(1).replace("$", "").replace(",", "")
    try:
        return float(x)
    except ValueError:
        return np.nan


def match_value(opts, x, tol=0.0051):
    """(pred, status) for the option(s) numerically equal to x."""
    if x is None or not np.isfinite(x):
        return None, "unparsed"
    v = np.array([opt_val(o) for o in opts], float)
    hit = np.flatnonzero(np.abs(v - x) <= tol + 1e-9 * abs(x))
    if len(hit) == 1:
        return int(hit[0]), "ok"
    if len(hit) == 0:
        return None, "no_match"
    if len({opts[i] for i in hit}) == 1:  # same text twice: still one answer
        return int(hit[0]), "ok"
    return None, "multi_match"


def match_text(opts, t):
    hit = [i for i, o in enumerate(opts) if str(o).strip().rstrip(".").lower() == t.lower()]
    if len(hit) == 1:
        return hit[0], "ok"
    return (None, "no_match") if not hit else (hit[0], "ok") if len({opts[i] for i in hit}) == 1 else (None, "multi_match")


# signed numbers as they appear in the arithmetic templates: 74, -74, $74, -$74, $-74, -$ 74, $(-74), 1,000
NUM = re.compile(r"(?<![\w.])(-)?(?:\$\s?)?\(?(-)?\$?(\d[\d,]*(?:\.\d+)?)\)?")


def signed_numbers(text: str):
    vals = []
    for m in NUM.finditer(text):
        v = float(m.group(3).replace(",", ""))
        vals.append(-v if (m.group(1) or m.group(2)) else v)
    i = [0]

    def rep(m):
        s = f"<N{i[0]}>"
        i[0] += 1
        return s
    return vals, NUM.sub(rep, text)


# =============================================================================================== add_sub
_AS_EXPLICIT = [
    (re.compile(r"subtract(?:ing|ed)?\s+<N0>\s+from\s+<N1>", re.I), lambda a, b: b - a),
    (re.compile(r"<N0>\s+(?:is\s+)?subtracted\s+from\s+<N1>", re.I), lambda a, b: b - a),
    (re.compile(r"<N0>\s*(?:-|−|–|minus)\s*<N1>", re.I), lambda a, b: a - b),
    (re.compile(r"<N0>\s*(?:\+|plus)\s*<N1>", re.I), lambda a, b: a + b),
    (re.compile(r"\b(?:sum|total) of\s+<N0>\s+and\s+<N1>", re.I), lambda a, b: a + b),
    (re.compile(r"\badd(?:ing)?\s+<N0>\s+(?:and|to|with)\s+<N1>", re.I), lambda a, b: a + b),
]


def _explicit(t, pats, a, b):
    for p, f in pats:
        if p.search(t):
            return f(a, b)
    return None


def solve_add_sub(df):
    preds, stats = [], []
    for qt, opts, tags in zip(df.question_text, df.options, df.tags):
        nums, t = signed_numbers(str(qt))
        tag = (list(tags) or ["?"])[0]
        if len(nums) != 2:
            preds.append(None), stats.append("unverified:extra_numbers")
            continue
        a, b = nums
        x = _explicit(t, _AS_EXPLICIT, a, b)
        if x is None:
            if tag == "add" and a >= 0 and b >= 0:  # validated family: 'add' story problems are plain sums
                x = a + b
            else:
                preds.append(None), stats.append(f"unverified:story_{tag}")
                continue
        p, s = match_value(list(opts), x)
        preds.append(p), stats.append(s)
    return _out(df.index, preds, stats)


# =============================================================================================== mult_div
_MD_EXPLICIT = [
    (re.compile(r"how many times (?:can|does|will|would|could)\s+<N0>\s+(?:go|fit)\s+into\s+<N1>", re.I),
     lambda a, b: b / a),
    (re.compile(r"<N0>\s+(?:go|goes|fit|fits)\s+into\s+<N1>", re.I), lambda a, b: b / a),
    (re.compile(r"<N0>\s+(?:is\s+)?divided\s+by\s+<N1>", re.I), lambda a, b: a / b),
    (re.compile(r"\bdivid(?:e|ing)\s+<N0>\s+by\s+<N1>", re.I), lambda a, b: a / b),
    (re.compile(r"<N0>\s*(?:/|÷)\s*<N1>", re.I), lambda a, b: a / b),
    (re.compile(r"\bquotient of\s+<N0>\s+(?:and|by|divided by)\s+<N1>", re.I), lambda a, b: a / b),
    (re.compile(r"\bdivision of\s+<N0>\s+by\s+<N1>", re.I), lambda a, b: a / b),
    (re.compile(r"<N0>\s*(?:\*|×|\bx\b|times|multiplied by)\s*<N1>", re.I), lambda a, b: a * b),
    (re.compile(r"\bmultipl(?:y|ying)\s+<N0>\s+(?:by|and|with|times)\s+<N1>", re.I), lambda a, b: a * b),
    (re.compile(r"\b(?:product|multiplication(?: result)?) of\s+<N0>\s+and\s+<N1>", re.I), lambda a, b: a * b),
]


def solve_mult_div(df):
    preds, stats = [], []
    for qt, opts, tags, ty in zip(df.question_text, df.options, df.tags, df.type):
        nums, t = signed_numbers(str(qt))
        tag = (list(tags) or ["?"])[0]
        if len(nums) != 2:
            preds.append(None), stats.append("unverified:extra_numbers")
            continue
        a, b = nums
        try:
            x = _explicit(t, _MD_EXPLICIT, a, b)
        except ZeroDivisionError:
            preds.append(None), stats.append("ill_posed:division_by_zero")
            continue
        if x is None:
            if tag == "multiply" and a > 0 and b > 0:  # validated family: positive multiplication stories
                x = a * b
            else:
                preds.append(None), stats.append(f"unverified:story_{tag}_{ty}")
                continue
        p, s = match_value(list(opts), round(x, 2))
        preds.append(p), stats.append(s)
    return _out(df.index, preds, stats)


def fix_mult_div(df):
    df = df.copy()
    # (1) unrounded division answers (positive/decimal: 15,845 rows) -> 2 decimals like the distractors
    def rnd(o):
        out = []
        for x in o:
            if re.fullmatch(r"-?\d+\.\d{3,}", x):
                out.append(str(round(float(x), 2)))
            else:
                out.append(x)
        return out
    new = df.options.map(lambda o: rnd(list(o)))
    ch = [list(a) != list(b) for a, b in zip(new, df.options)]
    df["options"] = new
    _add_repair(df, ch, "option_rounded")
    # (2) negative*negative stored as -(ab): negate every option (distractors are stored +- integer offsets)
    tags = df.tags_raw.fillna("")
    cand = (tags.str.startswith("multiply")) & (df.type == "negative")
    flip = np.zeros(len(df), bool)
    for k, (i, qt, opts, cor) in enumerate(zip(df.index, df.question_text, df.options, df.correct)):
        if not cand.loc[i]:
            continue
        nums, _ = signed_numbers(str(qt))
        w = np.flatnonzero(np.asarray(cor, float) == 1)
        if len(nums) != 2 or len(w) != 1 or nums[0] >= 0 or nums[1] >= 0:
            continue
        st = opt_val(opts[w[0]])
        if abs(st + nums[0] * nums[1]) <= 0.0051 + 1e-9 * abs(st):
            flip[k] = True
    neg = lambda s: (s[1:] if s.startswith("-") else "-" + s)
    df.loc[flip, "options"] = df.loc[flip, "options"].map(lambda o: [neg(x) for x in o])
    _add_repair(df, flip, "answer_sign_fixed")
    return df


# =============================================================================================== compute_expectations
_PTS = re.compile(r"(\d+(?:\.\d+)?) points")


def _ev(q: str):
    """Expected value from the compute_expectations templates; (value, status)."""
    s = q.strip()
    m = re.search(r"divided into (\d+) equal sections numbered (\d+) to (\d+)", s)
    if m:
        n, a, b = map(int, m.groups())
        if b - a + 1 != n:
            return None, "ill_posed:sections_mismatch"
        return (a + b) / 2, "ok"
    m = re.search(r"fair (\d+)-sided die is rolled once", s)
    if m:
        n = int(m.group(1))
        return (n + 1) / 2, "ok"
    m = re.search(r"(\d+)% (?:chance|likelihood) of [^.]*? and a (\d+)% (?:chance|likelihood) of .*?(?:for|over) (\d+) days", s)
    if m:
        p1, p2, d = map(int, m.groups())
        if p1 + p2 != 100:
            return None, "ill_posed:probabilities_not_summing_to_1"
        return p1 / 100 * d, "ok"
    m = re.search(r"contains (\d+) (\w+), (\d+) (\w+), and (\d+) (\w+).*?to get an? (\w+)", s)
    if m:
        g = m.groups()
        counts = {g[1]: int(g[0]), g[3]: int(g[2]), g[5]: int(g[4])}
        tgt = g[6].lower()
        hit = [c for w, c in counts.items() if w.lower().rstrip("s") == tgt.rstrip("s")]
        if len(hit) != 1 or hit[0] == 0:
            return None, "unparsed"
        return sum(counts.values()) / hit[0], "ok"
    pts = [float(x) for x in _PTS.findall(s)]
    if len(pts) == 2:
        win, lose = pts
        m = re.search(r"any of ((?:\d+[\s,]*(?:and\s+)?)+)", s)
        if m:
            faces = [int(x) for x in re.findall(r"\d+", m.group(1))]
            if len(set(faces)) != len(faces) or not all(1 <= f <= 6 for f in faces):
                return None, "ill_posed:bad_faces"
            k = len(faces)
            return (k * win - (6 - k) * lose) / 6, "ok"
        if re.search(r"\bcoin\b|\bdie\b|\bdice\b", s):
            return 0.5 * win - 0.5 * lose, "ok"
    return None, "unparsed"


def solve_compute_expectations(df):
    preds, stats = [], []
    for qt, opts in zip(df.question_text, df.options):
        x, st = _ev(str(qt))
        if st != "ok":
            preds.append(None), stats.append(st)
            continue
        p, s = match_value(list(opts), round(x, 2), tol=0.0051)
        preds.append(p), stats.append(s)
    return _out(df.index, preds, stats)


def _format_opts(opts, unit_fmt):
    out = []
    for o in opts:
        v = opt_val(o)
        out.append(o if not np.isfinite(v) else unit_fmt(v))
    return out


def fix_compute_expectations(df):
    """Correct options lack the unit (' points', 4,000 rows) and carry 1 decimal where distractors carry 2:
    every option is rewritten as '%.2f' + the row's unit."""
    df = df.copy()
    new = []
    for opts in df.options:
        units = [m.group(2) for m in (_UNIT.match(str(o)) for o in opts) if m and m.group(2)]
        u = max(set(units), key=units.count) if units else ""
        new.append(_format_opts(list(opts), lambda v, u=u: f"{v:.2f}" + (f" {u}" if u else "")))
    ch = [list(a) != list(b) for a, b in zip(new, df.options)]
    df["options"] = new
    _add_repair(df, ch, "option_format")
    return df


# =============================================================================================== bayes_rule
_PCT = re.compile(r"(\d+(?:\.\d+)?)%")


def solve_bayes_rule(df):
    preds, stats = [], []
    for qt, opts in zip(df.question_text, df.options):
        s = str(qt)
        n = [float(x) / 100 for x in _PCT.findall(s)]
        if len(n) != 3:
            preds.append(None), stats.append("unparsed")
            continue
        if re.match(r"\s*(When it|There is a \d+% chance that a new strain)", s):
            like, prior, ev = n  # P(E|H), P(H), P(E)   (few-shot examples 1_0, 2_0, 3_0)
        else:
            ev, like, prior = n  # generated: P(E), P(E|H), P(H) ('leg cancer' example: same product)
        if ev <= 0:
            preds.append(None), stats.append("ill_posed:zero_evidence")
            continue
        post = like * prior / ev
        if post > 1 + 1e-9:
            preds.append(None), stats.append("ill_posed:posterior_gt_1")
            continue
        p, st = match_value(list(opts), round(post * 100, 2), tol=0.0051)
        preds.append(p), stats.append(st)
    return _out(df.index, preds, stats)


def fix_bayes_rule(df):
    """domain cycles medical/weather/sports by row position although every generated question is a sports
    question; derive it from the text. Options normalised to 2 decimals (correct had 1 decimal twice as often)."""
    df = df.copy()
    t = df.question_text.str.lower()
    dom = np.where(t.str.contains("patient|cancer|influenza|disease"), "bayes_medical",
                   np.where(t.str.contains(r"sky|snow"), "bayes_weather", "bayes_sports"))
    ch = df.domain.to_numpy() != dom
    df["domain"] = dom
    _add_repair(df, ch, "meta_derived")
    new = [_format_opts(list(o), lambda v: f"{v:.2f}%") for o in df.options]
    ch = [list(a) != list(b) for a, b in zip(new, df.options)]
    df["options"] = new
    _add_repair(df, ch, "option_format")
    return df


# =============================================================================================== complement_rule
# templates whose stored answer 1 - p1 - p2 is not the answer to the question as written
_CR_ILL = [
    (r"For a fair dice roll, there is a", "non_disjoint"),                       # odd & <3 overlap; fair die fixed
    (r"standard deck\. The probability of drawing an ace", "non_disjoint"),     # ace & spade overlap
    (r"chance of drawing a heart, and a [\d.]+ chance of drawing a queen", "non_disjoint"),
    (r"the same house has a cat", "non_disjoint"),                              # dog & cat owners overlap
    (r"speak neither English nor Spanish", "non_disjoint"),                     # bilinguals
    (r"seeing neither a lion nor a bear", "non_disjoint"),
    (r"The probability \([\d.]+\) that you will roll an even number is", "garbled_numbers"),
    (r"The chance \([\d.]+\) that you will choose red is", "garbled_numbers"),
    (r"The chance \([\d.]+\) of you winning the jackpot is", "garbled_numbers"),
    (r"species B is never found with a probability", "garbled"),
    (r"three colors: gold, black, and white", "underdetermined"),               # asks white; black+white known
    (r"elephants, lions, or giraffes", "underdetermined"),
    (r"3 types of cards: hearts, diamonds, and clubs", "underdetermined"),
    (r"plays no sports at all", "underdetermined"),
    (r"three-colored dice with an", "underdetermined"),
    (r"awaiting for either 3, 5 or 2", "underdetermined"),                       # six-sided die, 3 faces given
]
_CR_ILL = [(re.compile(p), r) for p, r in _CR_ILL]
_PROB = re.compile(r"(?<![\d.,])(\d?\.\d+)(?![\d%])")


def solve_complement_rule(df):
    preds, stats = [], []
    for qt, opts in zip(df.question_text, df.options):
        s = str(qt)
        bad = next((r for p, r in _CR_ILL if p.search(s)), None)
        if bad:
            preds.append(None), stats.append(f"ill_posed:{bad}")
            continue
        ps = [float(x) for x in _PROB.findall(s)]
        if not ps or len(ps) > 2:
            preds.append(None), stats.append("unparsed")
            continue
        x = 1 - sum(ps)
        if x < -1e-9:
            preds.append(None), stats.append("ill_posed:negative_probability")
            continue
        p, st = match_value(list(opts), round(x, 2), tol=0.0051)
        preds.append(p), stats.append(st)
    return _out(df.index, preds, stats)


def fix_complement_rule(df):
    """'A fair coin is tossed. The probability of it showing heads is 0.57' contradicts itself (100 rows)."""
    df = df.copy()
    m = df.question_text.str.contains("A fair coin is tossed. The probability of it showing heads is", regex=False)
    df.loc[m, "question_text"] = df.loc[m, "question_text"].str.replace("A fair coin is tossed.",
                                                                        "A biased coin is tossed.", regex=False)
    _add_repair(df, m, "text_fixed")
    return df


# =============================================================================================== gamblers_fallacy
_W = r"(?:\w+)"
_GF_ILL = [
    # cards drawn without replacement: the next draw IS more/less likely (deck depletion) -> 'No' is wrong
    (r"draws a black card .* from a standard deck", "without_replacement"),
    (r"After drawing a heart suit from a deck", "without_replacement"),
    (r"blackjack table, the dealer has drawn a face card", "without_replacement"),
    (r"Blackjack table, the dealer has pulled out an Ace", "without_replacement"),
    (r"playing blackjack, and in .* consecutive deals, an ace", "without_replacement"),
    (r"When drawing cards from a well-shuffled deck", "without_replacement"),
    (r"While dealing cards, the last", "without_replacement"),
    (r"playing blackjack, the dealer has drawn a king", "without_replacement"),
    (r"blackjack table and the next card drawn has been a face card", "without_replacement"),
    (r"blackjack table, and dealer has drawn an Ace", "without_replacement"),
    (r"You draw a card from a standard deck and it", "without_replacement"),
    (r"pulled a non-winning card from a deck", "without_replacement"),
    (r"You pull a card from a shuffled standard deck", "without_replacement"),
    (r"blackjack table and notice the dealer has drawn a face card", "without_replacement"),
    (r"drawing cards from a (?:shuffled|well-mixed|well-shuffled) deck", "without_replacement"),
    (r"playing blackjack and the dealer has drawn a face card", "without_replacement"),
    (r"the dealer has had blackjack", "without_replacement"),
    (r"dealt a face-down card", "not_random"),                   # dealing procedure, not chance
    # not independent draws of a chance device
    (r"Rock, Paper, Scissors|rock-paper-scissors", "not_random"),  # human opponent
    (r"baseball game where the home team", "not_random"),
    (r"Russian Roulette", "not_random"),                         # without re-spin the hazard rises
    (r"playing bingo", "without_replacement"),                   # bingo numbers are not replaced
    # not a yes/no question
    (r"which color - red or black", "not_yes_no"),
    (r"another number or the same", "not_yes_no"),
    (r"more likely or not that", "not_yes_no"),
    # literal reading makes 'Yes' correct (absolute, not 'than before')
    (r"will the next roll likely not be a", "literal_yes"),
    (r"the next roll will probably not be a", "literal_yes"),
    (r"expect the next roll to be any other number than", "literal_yes"),
    (r"more likely to be another number than", "literal_yes"),
    (r"more likely to land on a number less than or equal to", "literal_yes"),  # 0..18 is 19/37 > 18/37
]
_GF_ILL = [(re.compile(p, re.I), r) for p, r in _GF_ILL]
# comparatives without 'than before' where 'other number' literally is more likely (kept; open question)
GF_AMBIGUOUS = re.compile(r"different number|another number|other number|something other than|anything but|"
                          r"different outcome|not show cherries|more likely not be|something else", re.I)


def solve_gamblers_fallacy(df):
    preds, stats = [], []
    for qt, opts in zip(df.question_text, df.options):
        s = str(qt)
        bad = next((r for p, r in _GF_ILL if p.search(s)), None)
        if bad:
            preds.append(None), stats.append(f"ill_posed:{bad}")
            continue
        p, st = match_text(list(opts), "No")
        preds.append(p), stats.append(st)
    return _out(df.index, preds, stats)


def fix_gamblers_fallacy(df):
    """'Yes.' (with period) vs 'No' (without) marks the correct option; strip the trailing period."""
    df = df.copy()
    new = df.options.map(lambda o: [x.strip().rstrip(".") if x.strip().rstrip(".") in ("Yes", "No") else x for x in o])
    ch = [list(a) != list(b) for a, b in zip(new, df.options)]
    df["options"] = new
    _add_repair(df, ch, "option_format")
    return df


# =============================================================================================== prob_basic
# Every one of the 868 rows was read and judged by hand (2026-09-27). base_id -> correct option text for
# stored answers that are wrong (49 rows):
PB_FIX = {
    "21~1": "The probability of pulling out a red card from a standard deck is 20%.",
    "26~1": "Drawing a number card",
    "33~1": "Two participants submitting a picture of a sunset with a 30% probability",
    "40~1": "Around 17%",                    # 1 - 365!/(353! 365^12) = 0.167
    "72": "Yes", "84": "Event A", "115": "Yes", "117": "No",
    "219": "Jane draws a blue marble", "222": "The card is an Ace", "223": "Sam picks a lemon candy",
    "226": "Drawing an Ace.", "248": "Rolling a 5",
    "284": "A coin landing heads up, if it has a 40% chance of heads",
    "304": "Yes", "319": "No", "334": "Yes", "345": "No", "360": "No", "375": "0.625", "376": "Event A",
    "388": "Yes", "396": "No", "450": "25%", "502": "Yes", "577": "No",
    "594": "Horse with 2 in 9 probability", "604": "The coin lands on heads", "610": "50%",
    "615": "Drawing a heart card from a well-shuffled deck with a 25% chance", "630": "Drawing a blue chip",
    "639": "No", "660": "A coin flip landing on heads", "664": "A concert event featuring a local band",
    "670": "It rains in the desert with a 5% chance",
    "671": "A person getting influenced by a TV commercial with about 30% chance",
    "680": "Getting a number less than 7", "683": "Picking a month with 31 days",
    "685": "Picking a person under 6 feet tall", "690": "Drawing a Heart", "692": "Rolling a 6 on a fair dice",
    "715": "Yes", "761": "Red area, with a probability of landing of 0.2",
    "765": "Box A is said to contain the prize with a probability of 0.6", "773": "The person has brown hair",
    "783": "Staff accident: 1 in 100", "804": "Yes", "817": "Yes",
    "544": "Yes",                            # 0 correct in the release; P(head)=P(tail)=0.5
}
# rows dropped: base_id -> reason
PB_DROP = {
    **{b: "no_correct_option" for b in ["64", "67", "170", "601", "699", "728"]},
    **{b: "multi_correct" for b in ["255", "256", "730", "749", "751"]},
    **{b: "tie" for b in ["27~1", "38~1", "247", "273", "771", "326", "251", "697", "695", "696"]},
    **{b: "no_information" for b in ["30~1", "48", "138", "140", "141", "142", "143", "341", "349", "535", "587",
                                     "806", "807", "809", "811", "813", "568", "221", "602"]},
    **{b: "no_question" for b in ["50", "626", "627", "633", "634", "656", "657", "658", "659", "662", "667", "693",
                                  "694", "777"]},
    **{b: "garbled" for b in ["25~1", "678", "687", "689", "691", "792", "81"]},
    **{b: "contradictory_premise" for b in ["199", "285", "286", "288", "605", "614", "617", "619"]},
    **{b: "ambiguous" for b in ["114", "271", "289", "801", "146", "227", "303", "448", "731", "793"]},
}
_MARK = " (Correct Option)"


def fix_prob_basic(df):
    df = df.copy()
    base = df.base_id.astype(str)
    # 1. '(Correct Option)' markers in the unanswered rows 305-314, 595-602 (18 rows): the marker is the answer
    for i in df.index[df.options.map(lambda o: any(_MARK in x for x in o))]:
        o = list(df.at[i, "options"])
        w = [k for k, x in enumerate(o) if _MARK in x]
        o = [x.replace(_MARK, "") for x in o]
        if base[i] == "309":
            o = [x.replace("7.6%", "7.69%") for x in o]           # 4/52 = 7.69%
        if base[i] == "312":
            o = ["Equal", "More likely to draw black", "More likely to draw red"]
            w = [0]
        df.at[i, "options"] = o
        df.at[i, "correct"] = [1.0 if k == w[0] else 0.0 for k in range(len(o))] if len(w) == 1 else df.at[i, "correct"]
        df.at[i, "repair"] = list(df.at[i, "repair"]) + ["option_marker_removed"] + (
            ["text_fixed"] if base[i] in ("309", "312") else [])
    # 2. 543: options contain fragments of question 544 ('mple 39:', 'sidering a coin toss event ...')
    for i in df.index[base == "543"]:
        df.at[i, "options"] = ["Yes", "No"]
        df.at[i, "correct"] = [1.0, 0.0]
        df.at[i, "repair"] = list(df.at[i, "repair"]) + ["option_garbage_removed"]
    # 3. wrong stored answers
    for i in df.index[base.isin(PB_FIX)]:
        o = list(df.at[i, "options"])
        tgt = PB_FIX[base[i]]
        w = [k for k, x in enumerate(o) if x.strip() == tgt]
        if len(w) != 1:
            raise RuntimeError(f"prob_basic {base[i]}: option {tgt!r} not found in {o}")
        df.at[i, "correct"] = [1.0 if k == w[0] else 0.0 for k in range(len(o))]
        df.at[i, "repair"] = list(df.at[i, "repair"]) + ["answer_manual"]
    # 4. drops
    m = base.isin(PB_DROP)
    df.loc[m, "drop"] = "manual:" + base[m].map(PB_DROP)
    return df


# =============================================================================================== registry
_VERIFIED = ["ill_posed", "no_match", "multi_match", "unverified", "unparsed"]
SOLVERS = {
    "add_sub": solve_add_sub,
    "mult_div": solve_mult_div,
    "compute_expectations": solve_compute_expectations,
    "bayes_rule": solve_bayes_rule,
    "complement_rule": solve_complement_rule,
    "gamblers_fallacy": solve_gamblers_fallacy,
}
POLICY = {
    "add_sub": {"grading": "answer", "authoritative": True, "drop_status": _VERIFIED, "shuffle": None},
    "mult_div": {"grading": "answer", "authoritative": True, "drop_status": _VERIFIED, "shuffle": None},
    "compute_expectations": {"grading": "answer", "authoritative": True,
                             "drop_status": ["ill_posed", "no_match", "multi_match"], "shuffle": None},
    "bayes_rule": {"grading": "answer", "authoritative": True,
                   "drop_status": ["ill_posed", "no_match", "multi_match"], "shuffle": None},
    "complement_rule": {"grading": "answer", "authoritative": True,
                        "drop_status": ["ill_posed", "no_match", "multi_match"], "shuffle": None},
    "gamblers_fallacy": {"grading": "answer", "authoritative": True, "drop_status": ["ill_posed"], "shuffle": None},
    "prob_basic": {"grading": "answer", "authoritative": False, "drop_status": [], "shuffle": None},
}
FIXES = {
    "mult_div": fix_mult_div,
    "compute_expectations": fix_compute_expectations,
    "bayes_rule": fix_bayes_rule,
    "complement_rule": fix_complement_rule,
    "gamblers_fallacy": fix_gamblers_fallacy,
    "prob_basic": fix_prob_basic,
}

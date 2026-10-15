"""Fillers for the STEER arithmetic elements add_sub and mult_div (rebuilt 2026-09; the original generator is lost).

Templates were recovered from the released questions and labelled by hand with the operation each story implies
(audit/steer/regen/recover_arith_templates.py). The labels live next to the templates in
data/templates/<element>/template_labels.csv and are looked up by the template text the sampler passes in
metadata['question_text']. The answer is computed exactly (fractions) from the numbers shown in the question and
rounded to 2 decimals; every option uses the same rounding.

Numbers by difficulty (the released grade structure: more digits = harder):
    0  integers 2-99
    1  2-decimal numbers in [1, 100) (count slots: integers 2-99)
    2  2-decimal numbers in [1000, 10000) (count slots: integers 1000-9999)
add_sub: type integer/decimal/large = difficulty 0/1/2; the numbers are magnitudes, signs written in the story
("debt of -${a}") are part of the template. mult_div: type positive/negative/both = signs of the two numbers
(negative/both only for explicit-arithmetic templates).

Options: the answer x, one digit slip of x, a wrong-operation value w (or, for explicit arithmetic only, the sign
error -x) and one digit slip of w. The two pairs are built the same way, so neither the correct pair nor the
correct member of a pair can be read off the option set. -x is offered only for explicit arithmetic with a negative
number or a negative answer (stories: the sign convention is not always unambiguous; all-positive sums or products:
not a plausible slip), and then options may have either sign; otherwise all options share the answer's sign. A 0.01
slip is not used next to a rounded (inexact) answer. mult_div (October 2026): every option is within a factor 3 of
the answer in size, so a wrong-operation value is used only when it is of the answer's order of magnitude (a quotient
offered for a product of four-digit numbers could be ruled out by its size); otherwise the second pair starts from a
slip in one of the answer's two leading digits. Non-exact mult_div questions state the rounding (ROUNDING_NOTE). add_sub
story templates whose negative amounts conflict with what they ask are dropped (SIGNED_STORY_KEEP).
"""
from __future__ import annotations

import csv
import os
import re
from fractions import Fraction
from functools import lru_cache
from math import ceil, floor

from .fill_registry import FillRegister
from .utilities.filler_utils import rng

function_registry = FillRegister()

from autosteer.paths import templates_dir  # the data directory, resolved when the labels are read
ADD_SUB_LEVEL = {'integer': 0, 'decimal': 1, 'large': 2}
NEG_SLOT = re.compile(r'-\$?\s?\{[ab]\}|\$-\{[ab]\}')  # a number written negative in the template

AS_OPS = {
    'a+b': lambda a, b: a + b,
    'a-b': lambda a, b: a - b,
    'b-a': lambda a, b: b - a,
    '-a-b': lambda a, b: -a - b,
}
MD_OPS = {
    'a*b': lambda a, b: a * b,
    'a/b': lambda a, b: a / b,
    'b/a': lambda a, b: b / a,
}


@lru_cache(maxsize=None)
def labels(element: str) -> dict[str, dict]:
    """Kept templates of `element`: template text -> label row."""
    path = os.path.join(templates_dir(), element, 'template_labels.csv')
    with open(path, newline='') as f:
        return {r['template']: r for r in csv.DictReader(f) if r['status'] == 'keep'}


# mult_div: every story or explicit template that is not an exact division asks for a rounded number; the
# convention is stated in the question (stated in the stored templates, stripped again here so the
# label lookup sees the template text of the data directory).
ROUNDING_NOTE = ' If the result is not a whole number, round it to two decimal places.'

# add_sub: story templates that write the amounts as negative numbers ("descends -{a} feet", "an overdraft of -{a}")
# and then ask for a magnitude ("how far", "how much do you owe") have no consistent reading: the keyed signed sum
# is not the distance or amount asked for. Only the stories below, which ask for a signed quantity (a balance, a
# score, an elevation change) or state the sign convention, are kept; the other story templates with a
# negative slot are dropped from the stored templates. Explicit arithmetic ("-69 - 11") is unaffected.
SIGNED_STORY_KEEP = re.compile(
    r'think of your debt as negative numbers|start with negative points|bank balance which is -\{a\}'
    r'|overdrawn by -\{a\} dollars\. You deposit|elevation of -\{a\} feet')


def keep_add_sub_template(text: str) -> bool:
    """False for an add_sub story template whose negative amounts conflict with what it asks (see above)."""
    lab = labels('add_sub').get(text)
    if lab is None or 'E' in (lab['flags'] or '') or not NEG_SLOT.search(text):
        return True
    return bool(SIGNED_STORY_KEEP.search(text))


# mult_div wordings where a slot reads as a total although the story multiplies it (data text, shown text)
MULT_DIV_REWORDS = [
    ('Your coffee cost {a} and you bought {b} cups this week.', 'Each cup of coffee cost {a} and you bought {b} cups this week.'),
]


def with_rounding_note(text: str) -> str:
    """mult_div template text with the rounding convention (not for exact divisions, whose answer is whole), and
    the rewordings of MULT_DIV_REWORDS."""
    lab = labels('mult_div').get(text)
    for old, new in MULT_DIV_REWORDS:
        text = text.replace(old, new)
    if lab is None or 'X' in (lab['flags'] or '') or text.endswith(ROUNDING_NOTE):
        return text
    body = text.rstrip()
    if body[-1] not in '.?!"':  # "What is {a} multiplied by {b}", "Solve the following division: {a} / {b}"
        last = re.split(r'(?<=[.?!])\s+', body)[-1]
        body += '?' if re.match(r'(What|How|Which)\b', last) else '.'
    return body + ROUNDING_NOTE


def base_template(element: str, text: str | None) -> str | None:
    """The data directory's template text of a (load-time corrected) template text: without the rounding note and
    the punctuation added before it."""
    if text and text.endswith(ROUNDING_NOTE):
        text = text[:-len(ROUNDING_NOTE)]
        if text not in labels(element) and text[:-1] in labels(element):
            text = text[:-1]
    if text and element == 'mult_div':
        for old, new in MULT_DIV_REWORDS:
            text = text.replace(new, old)
    return text


def _label(element: str, metadata: dict) -> dict:
    text = base_template(element, metadata.get('question_text'))
    lab = labels(element).get(text) if text else None
    if lab is None:
        if text:
            raise KeyError(f'{element}: template has no operation label: {text[:120]!r}')
        # called without a template (the sampler probing which keys the filler returns for a metadata
        # combination): an explicit-arithmetic label, which is valid for every type
        lab = {'op': 'a+b' if element == 'add_sub' else 'a*b', 'flags': 'E', 'money_a': '', 'money_b': '',
               'template': ''}
    return lab


# ------------------------------------------------------------------------------------------------ numbers
def _uniform_cents(lo: int, hi: int) -> Fraction:
    """2-decimal number in [lo, hi) whose last digit is not 0 (so the shown text keeps both decimals)."""
    while True:
        c = int(rng.integers(lo * 100, hi * 100))
        if c % 10:
            return Fraction(c, 100)


def magnitude(level: int, count: bool) -> Fraction:
    if level == 0 or (level == 1 and count):
        return Fraction(int(rng.integers(2, 100)))
    if level == 1:
        return _uniform_cents(1, 100)
    if count:
        return Fraction(int(rng.integers(1000, 10000)))
    return _uniform_cents(1000, 10000)


def round2(x: Fraction) -> Fraction:
    """Round half away from zero to 2 decimals."""
    s = 1 if x >= 0 else -1
    return s * Fraction(floor(abs(x) * 100 + Fraction(1, 2)), 100)


def on_half(x: Fraction) -> bool:
    """x lies exactly on a rounding boundary at 2 decimals (rounding conventions would disagree)."""
    return (abs(x) * 100) % 1 == Fraction(1, 2)


def fmt(x: Fraction, money: bool = False) -> str:
    x = round2(x)
    s = str(int(x)) if x.denominator == 1 else f'{float(x):.2f}'.rstrip('0')
    if money:
        return ('-$' + s[1:]) if s.startswith('-') else '$' + s
    return s


# ------------------------------------------------------------------------------------------------ options
def _slip(v: Fraction, exact: bool) -> Fraction:
    """A one-digit slip of v: +-10^k for a digit position k of v (k >= -2, within the top 7 significant digits; 0.01 only if v is exact)."""
    digits = len(str(int(abs(v)))) if abs(v) >= 1 else 0
    ks = list(range(-2, max(digits, 1)))
    if not exact or v.denominator == 1:
        ks = [k for k in ks if k != -2]
    if v.denominator == 1:
        ks = [k for k in ks if k >= 0]
    ks = [k for k in ks if k >= digits - 7] or ks[-1:]  # within the top 7 significant digits
    k = ks[int(rng.integers(len(ks)))]
    step = Fraction(10) ** k
    return v + step if rng.random() < 0.5 else v - step


def _int_slip(v: Fraction) -> Fraction:
    """Off-by-one / off-by-two / off-by-ten slip of a whole-number answer."""
    return v + (1 if rng.random() < 0.5 else -1) * int(rng.choice([1, 2, 10]))


def _near(w: Fraction, x: Fraction, ratio: float) -> bool:
    """|w| within a factor `ratio` of |x| (a distractor of the answer's order of magnitude)."""
    return w != 0 and 1 / ratio <= abs(w) / abs(x) <= ratio


def build_options(x: Fraction, exact: bool, wrong: list[Fraction], allow_negation: bool,
                  same_sign: bool, integer: bool = False, max_ratio: float | None = None) -> list[Fraction] | None:
    """[x, slip(x), w, slip(w)], all distinct and non-zero after rounding, or None if no valid set was found.

    w is a wrong-operation value (or -x when allow_negation). Both slips follow the same rule (no 0.01 slip if the
    answer is rounded), so the pairs are indistinguishable by construction. same_sign: every option has the sign
    of x. integer: every option is a positive integer (answers that count whole things). max_ratio: every option
    is within this factor of x in size (a wrong-operation value of another order of magnitude, e.g. a quotient
    offered for a product, can be ruled out without computing; the fallback is then a carry error of x)."""
    cand = [w for w in wrong if round2(w) != round2(x) and round2(w) != 0]
    if max_ratio:
        cand = [w for w in cand if _near(round2(w), x, max_ratio)]
    if not allow_negation:
        cand = [w for w in cand if round2(w) != -round2(x)]
    if same_sign:
        cand = [w for w in cand if (w > 0) == (x > 0)]
    for _ in range(30):
        if allow_negation and rng.random() < 0.5:
            w = -x
        elif cand:
            w = cand[int(rng.integers(len(cand)))]
        else:  # no usable wrong-operation value: a larger slip of x (e.g. a carry error)
            n = len(str(int(abs(x)))) if abs(x) >= 1 else 0
            if max_ratio:  # a slip in one of the two leading digits
                k = max(n - 1 - int(rng.integers(0, 2)), -1 if abs(x) >= Fraction(1, 10) else -2)
            else:
                k = int(rng.integers(1, max(2, n)))
            w = x + (1 if rng.random() < 0.5 else -1) * Fraction(10) ** k
        if integer:
            opts = [x, _int_slip(x), w, _int_slip(w)]
        else:
            opts = [x, _slip(x, exact), w, _slip(w, exact)]
        r = [round2(o) for o in opts]
        if len({fmt(o) for o in r}) < 4 or any(o == 0 for o in r):
            continue
        if same_sign and any((o > 0) != (x > 0) for o in r):
            continue
        if max_ratio and not all(_near(o, x, max_ratio) for o in r[1:]):
            continue
        if not allow_negation and any(o == -r[0] for o in r[1:]):
            continue
        if integer and any(o.denominator != 1 or o <= 0 for o in r):
            continue
        return r
    return None


def _result(slots: dict, opts: list[Fraction]) -> dict:
    return {**slots, 'options_list': [[fmt(o) for o in opts]], 'correct_answer': [[1, 0, 0, 0]]}


# ------------------------------------------------------------------------------------------------ add_sub
@function_registry.register('add_sub')
def fill_add_sub(metadata: dict) -> dict:
    lab = _label('add_sub', metadata)
    op, flags = lab['op'], lab['flags'] or ''
    level = ADD_SUB_LEVEL.get(metadata.get('type'), int(metadata.get('difficulty_level', 0)))
    count = 'I' in flags
    explicit = 'E' in flags
    f = AS_OPS[op]
    for _ in range(500):
        a, b = magnitude(level, count), magnitude(level, count)
        if a == b:
            continue
        if 'G' in flags and a < b:
            a, b = b, a
        x = f(a, b)
        if 'P' in flags and x <= 0:
            a, b = b, a
            x = f(a, b)
            if x <= 0 or ('G' in flags and a < b):
                continue
        if x == 0:
            continue
        wrong = [g(a, b) for name, g in AS_OPS.items() if name != op]
        # the sign error -x is a plausible slip only where signs matter: a negative number in the expression or a
        # negative answer
        negation = explicit and (x < 0 or bool(NEG_SLOT.search(lab.get('template', ''))))
        opts = build_options(x, True, wrong, allow_negation=negation, same_sign=not negation)
        if opts is None:
            continue
        return _result({'a': fmt(a), 'b': fmt(b)}, opts)
    raise RuntimeError(f'add_sub: no valid question for {op} {flags} {metadata.get("type")}')


# ------------------------------------------------------------------------------------------------ mult_div
MD_RATIO = 3  # mult_div options are within a factor 3 of the answer in size (review, October 2026)


def _signed(a: Fraction, b: Fraction, sign_type: str) -> tuple[Fraction, Fraction]:
    if sign_type == 'negative':
        return -a, -b
    if sign_type == 'both':
        return (-a, b) if rng.random() < 0.5 else (a, -b)
    return a, b


def _exact_division(level: int, count_divisor: bool, count_dividend: bool) -> tuple[Fraction, Fraction]:
    """(dividend, divisor) with an integer quotient; dividend has the digits of `level`."""
    lo, hi = (1000, 9999) if level == 2 else (4, 99)
    for _ in range(200):
        if count_divisor or level == 0 or count_dividend:
            d = Fraction(int(rng.integers(2, 100 if level == 2 else 13)))
        elif level == 1:
            d = _uniform_cents(1, 20)
        else:
            d = _uniform_cents(10, 100)
        qlo, qhi = max(2, ceil(lo / d)), floor(hi / d)
        if qlo > qhi:
            continue
        q = int(rng.integers(qlo, qhi + 1))
        return d * q, d
    raise RuntimeError('mult_div: no exact division found')


@function_registry.register('mult_div')
def fill_mult_div(metadata: dict) -> dict:
    lab = _label('mult_div', metadata)
    op, flags = lab['op'], lab['flags'] or ''
    level = int(metadata.get('difficulty_level', 0))
    sign_type = metadata.get('type', 'positive')
    explicit = 'E' in flags
    if sign_type != 'positive' and not explicit:
        raise ValueError(f'mult_div: sign type {sign_type!r} is only defined for explicit-arithmetic templates')
    count_a, count_b = 'A' in flags, 'B' in flags
    money_a = lab.get('money_a') == 'True' and not count_a
    money_b = lab.get('money_b') == 'True' and not count_b
    f = MD_OPS[op]
    for _ in range(500):
        if 'X' in flags:
            if op == 'a/b':
                a, b = _exact_division(level, count_b, count_a)
            else:
                b, a = _exact_division(level, count_a, count_b)
        else:
            a, b = magnitude(level, count_a), magnitude(level, count_b)
        if a == b:
            continue
        a, b = _signed(a, b, sign_type)
        x_exact = f(a, b)
        if on_half(x_exact):
            continue
        x = round2(x_exact)
        if x == 0:
            continue
        exact = x == x_exact
        if 'X' in flags:
            wrong = [g(a, b) for name, g in MD_OPS.items() if name != op]
            wrong = [w for w in wrong if w.denominator == 1]
            opts = build_options(x, True, wrong, allow_negation=False, same_sign=True, integer=True,
                                 max_ratio=MD_RATIO)
        else:
            wrong = [round2(g(a, b)) for name, g in MD_OPS.items() if name != op]
            negation = explicit and (a < 0 or b < 0)  # sign rules only matter with a negative factor
            if negation:
                wrong += [-w for w in wrong]
            opts = build_options(x, exact, wrong, allow_negation=negation, same_sign=not negation,
                                 max_ratio=MD_RATIO)
        if opts is None:
            continue
        return _result({'a': fmt(a, money_a), 'b': fmt(b, money_b)}, opts)
    raise RuntimeError(f'mult_div: no valid question for {op} {flags} {sign_type} {level}')

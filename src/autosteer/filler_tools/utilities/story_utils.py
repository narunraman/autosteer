"""Shared pieces of the hand-written (code) templates: perspectives and distractor selection.

Perspectives (template tags) as in factor_demand: first_person, second_person, man, woman, anon. `voice` gives the
subject forms a template sentence needs; `pick_apart` chooses mistake-based distractors that are kept apart from the
key and from each other.
"""
from __future__ import annotations

import math
import re

MEN = ['Daniel', 'Omar', 'Kenji', 'Lucas', 'Samuel', 'Mateo', 'Arjun']
WOMEN = ['Maria', 'Priya', 'Grace', 'Sofia', 'Hannah', 'Amara', 'Lena']
PERSPECTIVES = ('first_person', 'second_person', 'man', 'woman', 'anon')


def voice(tag: str, k: int, anon_subject: str, anon_possessive: str | None = None) -> dict:
    """Forms of the decision maker for a perspective.

    subj: sentence subject ("I", "You", "Maria", anon_subject); s: third-person verb suffix ('' or 's');
    be: "am" / "are" / "is"; poss: possessive ("my", "your", "Maria's", ...); obj: object ("me", "you", "her", ...);
    q: subject inside a question ("should I", "should you", "should Maria", ...); Subj is subj capitalised.
    """
    if tag == 'first_person':
        v = dict(subj='I', s='', be='am', poss='my', obj='me', q='I', have='have')
    elif tag == 'second_person':
        v = dict(subj='you', s='', be='are', poss='your', obj='you', q='you', have='have')
    elif tag in ('man', 'woman'):
        name = (MEN if tag == 'man' else WOMEN)[k % 7]
        v = dict(subj=name, s='s', be='is', poss=f"{name}'s", obj='him' if tag == 'man' else 'her', q=name,
                 have='has')
    elif tag == 'anon':
        v = dict(subj=anon_subject, s='s', be='is', poss=anon_possessive or f"{anon_subject}'s", obj=anon_subject,
                 q=anon_subject, have='has')
    else:
        raise ValueError(f'unknown perspective {tag!r}')
    v['Subj'] = v['subj'][0].upper() + v['subj'][1:]
    v['Poss'] = v['poss'][0].upper() + v['poss'][1:]
    return v


_SENTENCE = re.compile(r'[^.?!]*(?:[.?!]+(?=\s|$)|$)')
_POSS_PRONOUN = {**{n: 'his' for n in MEN}, **{n: 'her' for n in WOMEN}}
# lowercase 'a', or 'A' at the start of a sentence (elsewhere 'A' is a label: "Firm A is", "an A in the course")
_A_VOWEL = re.compile(r"(?:\b(a)|(?:(?<=^)|(?<=[.!?:]\s)|(?<=\n))(A)) (?=(?!one\b|once\b)[aeio][a-z])")


def articles(text: str) -> str:
    """'a' before a word starting with a, e, i or o becomes 'an' ("a online course provider" -> "an online ...").
    Words starting with u are left alone ("a utility", "a unit")."""
    return _A_VOWEL.sub(lambda m: (m.group(1) or m.group(2)) + 'n ', text)


def possessive_pronouns(text: str, others: dict[str, str] | None = None) -> str:
    """Within a sentence, the possessive of a perspective name (MEN / WOMEN) already mentioned earlier in the sentence
    becomes 'his' / 'her': "Maria invests Maria's savings" -> "Maria invests her savings", "If Grace's income rises,
    how does Grace's demand change" -> "... how does her demand change". `others` adds referents with their
    pronoun (e.g. {'the player': 'their'} for an anonymous perspective). Kept when another referent with the same
    pronoun was mentioned in between (the pronoun would be ambiguous). Anonymous subjects are not handled
    generically: a "the <noun> ... the <noun>'s" rule would rewrite logic clauses (equiv_contrapositive) and
    committee wording."""
    pronoun = {**_POSS_PRONOUN, **(others or {})}
    mention = re.compile(r"\b(" + '|'.join(re.escape(w) for w in sorted(pronoun, key=len, reverse=True))
                         + r")\b('s\b)?")

    def sentence(m):
        sent = m.group(0)
        out, pos, seen = [], 0, []
        for h in mention.finditer(sent):
            who, poss = h.group(1), bool(h.group(2))
            pron = pronoun[who]
            if poss and who in seen:
                last = len(seen) - 1 - seen[::-1].index(who)
                if not any(w != who and pronoun[w] == pron for w in seen[last + 1:]):
                    out.append(sent[pos:h.start()] + pron)
                    pos = h.end()
                    continue
            seen.append(who)
        out.append(sent[pos:])
        return ''.join(out)
    return _SENTENCE.sub(sentence, text)


def pick_apart(r, key: float, candidates, n: int = 3, rel: float = 0.1, abs_gap: float = 0.0,
               lo: float | None = None, hi: float | None = None, shown=None):
    """n candidates, each at least max(rel * larger magnitude, abs_gap) away from the key and from each other.

    Candidates outside [lo, hi] or not finite are dropped. `shown` (a formatter) is applied before the comparison,
    so that options that would print the same are never both chosen. Returns None when fewer than n qualify; the
    order of the candidates is randomised with `r`.
    """
    def apart(x, y):
        return abs(x - y) >= max(rel * max(abs(x), abs(y)), abs_gap) - 1e-12

    cands = [float(c) for c in candidates if c is not None and math.isfinite(c)
             and (lo is None or c >= lo) and (hi is None or c <= hi)]
    order = [int(i) for i in r.permutation(len(cands))]
    chosen: list[float] = []
    for i in order:
        c = cands[i]
        if all(apart(c, x) for x in [key] + chosen):
            if shown is not None and shown(c) in {shown(x) for x in [key] + chosen}:
                continue
            chosen.append(c)
            if len(chosen) == n:
                return chosen
    return None


def money(x: float) -> str:
    """$1,234.56"""
    return f'${x:,.2f}'


def num(x: float, places: int = 2) -> str:
    """A number rounded to `places` decimals, without trailing zeros (12.5, 7, 0.04)."""
    v = round(float(x), places)
    if v == 0:
        v = 0.0
    s = f'{v:.{places}f}'.rstrip('0').rstrip('.') if places else str(int(round(v)))
    return '0' if s in ('-0', '') else s


_DECIMAL = re.compile(r'\d+\.\d+')


def stable(text: str) -> bool:
    """True if the sampler's 2-decimal rounding of strings (dataset_filler.format_value) leaves `text` unchanged.

    It rewrites every decimal number with str(round(x, 2)), so "$12.30" would be shown as "$12.3"; fillers that
    show cents redraw values whose text would change.
    """
    return _DECIMAL.sub(lambda m: str(round(float(m.group(0)), 2)), text) == text


def dollars_amount(x: float) -> str:
    """An amount in dollars without the sign: whole dollars as "3", otherwise to the cent ("2.75")."""
    return f'{x:,.0f}' if abs(x - round(x)) < 1e-9 else f'{x:,.2f}'

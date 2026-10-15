"""fair_division (STEER 4.2.d, Fair Division Algorithms): hand-written templates.

Design of 6 October 2026 (the old notebook stub had one stem and no determinate keys). Each question names the
procedure and asks for its outcome on stated valuations; all valuations are additive.

Types:
    divider_chooser (difficulty 1): a cake (12 inches) or a strip of land (60 metres) is one kind (chocolate /
        orchard) up to a stated point and another kind after it; the divider and the chooser value each kind at a
        stated number of dollars per unit. The divider cuts where the two pieces are worth the same to them; the
        chooser takes the piece worth more to them. Three asks (spec 'ask'): where the cut is; which piece the
        chooser takes and what it is worth to the chooser; what the divider's piece is worth to the divider (half of
        the divider's total). Distractors: the midpoint, the chooser's own half-way point, the boundary, the other
        piece, the chooser's half, values by the other person's measure.
    last_diminisher (difficulty 2): three or four people, a 12-inch cake made of three flavour sections of stated
        lengths, each person's points for each section (total 60 each). In turn order, the first person cuts from the
        left end a piece worth exactly 1/n of the whole to them; each later person who thinks the piece is worth more
        than 1/n trims it to 1/n by their measure; the last person to cut or trim takes it; this repeats with the
        others and the rest of the cake, each piece worth 1/k of the rest by the cutter's or trimmer's measure when k
        people are left (the standard procedure; re-review, October 2026: it used 1/n of the whole cake in every
        round), and the last person takes what is left. Asks: who gets the first piece and
        where it ends (n = 3 or 4), or the final allocation (n = 3). Positions are shown rounded to 2 decimals; draws
        in which two people's cut points are within 0.1 inch are redrawn.
    adjusted_winner (difficulty 2): two people split 4 to 6 items with 100 points each (no item valued equally, all
        ratios distinct). Each item first goes to whoever values it more; the person ahead hands items over in
        increasing order of the ratio (own points / other's points) until the next item would put them behind; that
        item is split so both end with equal points. Options: which item is split and the share each gets (1
        decimal); distractors: the shares swapped, another item split, an even split.
    envy_check (difficulty 1): three people, 4 or 5 indivisible items, each person's values (100 in total each) and
        a proposed allocation of all items. Options: envy-free and proportional / proportional but not envy-free /
        neither (envy-free but not proportional is impossible when everything is allocated with additive values, so
        it is not offered); the three outcomes are equally common.
Perspectives: anon, first_person and second_person (the reader runs the division), ai_assistant.
"""
from __future__ import annotations

from fractions import Fraction as F
from itertools import permutations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import num

function_registry = FillRegister()

ELEMENT = 'fair_division'
PERSPECTIVES = ('anon', 'first_person', 'second_person', 'ai_assistant')
NAMES = ['Ana', 'Ben', 'Cara', 'Dev', 'Eli', 'Femi', 'Gus', 'Hana', 'Ivo', 'Jade', 'Kofi', 'Lena', 'Milo', 'Nina',
         'Omar', 'Pia', 'Raj', 'Sara', 'Tom', 'Uma']
# divisible goods: domain, thing, unit, length, first kind, second kind, who
LINEAR = [
    dict(domain='food', thing='a 12-inch sheet cake', short='cake', unit='inch', units='inches', length=12,
         kinds=('chocolate', 'vanilla', 'lemon'), part='section'),
    dict(domain='land', thing='a strip of land 60 metres long', short='strip', unit='metre', units='metres',
         length=60, kinds=('orchard', 'meadow', 'vegetable beds'), part='stretch'),
]
ITEMS = {
    'inheritance': ['the piano', 'the painting', 'the car', 'the watch', 'the stamp collection', 'the cottage'],
    'business': ['the client list', 'the office lease', 'the delivery van', 'the brand name', 'the patent portfolio',
                 'the equipment'],
}
# 'It is envy-free but not proportional' is not offered: with additive values and every item given out it is
# impossible, so it could be ruled out without looking at the allocation (review, October 2026).
EF_OPTIONS = ['It is both envy-free and proportional', 'It is proportional but not envy-free',
              'It is neither envy-free nor proportional']


def _lead(persp: str, who: str, what: str) -> str:
    """Opening sentence: who divides what, from a perspective."""
    if persp == 'anon':
        return f'{who} are dividing {what}.'
    if persp == 'first_person':
        return f'I am helping {who} divide {what}.'
    if persp == 'second_person':
        return f'You are helping {who} divide {what}.'
    return f'You are an AI assistant helping {who} divide {what}.'


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for persp in PERSPECTIVES:
        for li, g in enumerate(LINEAR):
            u = g['units']
            setup = (_lead(persp, '{who}', g['thing']) + f" Measured from the left end, the first {{a}} {u} are "
                     f"{g['kinds'][0]} and the rest is {g['kinds'][1]}. {{values}} They use divide and choose: "
                     f"{{divider}} cuts the {g['short']} into two pieces that are worth the same to {{divider}}, and "
                     f"{{chooser}} takes the piece that is worth more to {{chooser}}.")
            asks = {
                'cut': f" How far from the left end, in {u}, does {{divider}} cut?",
                'chooser': " Which piece does {chooser} take, and what is it worth to {chooser}?",
                'divider_share': " What is the piece {divider} ends up with worth to {divider}?",
            }
            for ask, q in asks.items():
                out.append(ct.CodeTemplate(parts=(setup + q,), type='divider_chooser', domain=g['domain'],
                                           tags=persp, difficulty_level=1,
                                           spec={'type': 'divider_chooser', 'g': li, 'ask': ask}))
            if li == 0:
                for ask, n in (('first', 3), ('first', 4), ('final', 3)):
                    setup = (_lead(persp, '{who}', g['thing']) + " Measured from the left end, the cake is {sections}. "
                             "Each person's points for the sections (" + str(n * 20) + " in total for each person) "
                             "are:\n{points}\nThey use the last diminisher procedure, in the turn order {order}: the "
                             f"first person cuts from the left end a piece worth exactly 1/{n} of the whole cake to "
                             f"them; each later person who thinks the piece is worth more than 1/{n} trims it to "
                             f"exactly 1/{n} by their own points; the last person to cut or trim takes the piece. "
                             "This repeats with the remaining people (in the same order) and the rest of the cake: "
                             "with k people left, the piece is worth exactly 1/k of the rest of the cake to its cutter "
                             "and to anyone who trims it. The last person takes whatever is left. Points are spread "
                             "evenly within each section.")
                    q = (" Who gets the first piece, and where does it end (in inches from the left end, rounded to "
                         "two decimals)?" if ask == 'first' else
                         " What is the final allocation (cut positions in inches from the left end, rounded to two "
                         "decimals)?")
                    out.append(ct.CodeTemplate(parts=(setup + q,), type='last_diminisher', domain='food', tags=persp,
                                               difficulty_level=2,
                                               spec={'type': 'last_diminisher', 'ask': ask, 'n': n}))
        for dom in ITEMS:
            what = 'the family estate' if dom == 'inheritance' else 'the assets of their former business'
            setup = (_lead(persp, '{who}', what) + " Each of them spreads 100 points over the items:\n{points}\n"
                     "They use the adjusted winner procedure: each item first goes to whoever gives it more points; "
                     "then the person with more points hands items over to the other, starting with the item for "
                     "which the ratio of their own points to the other's points is smallest, and splits the item at "
                     "which the totals would cross, so that both end with the same number of points.")
            out.append(ct.CodeTemplate(parts=(setup + " Which item is split, and how?",), type='adjusted_winner',
                                       domain=dom, tags=persp, difficulty_level=2, spec={'type': 'adjusted_winner'}))
            setup = (_lead(persp, '{who}', what) + " Each of them values the items as follows (100 points in total "
                     "for each person):\n{points}\nThe proposed allocation, which gives out every item, is: "
                     "{allocation}. An allocation is proportional if everyone gets at least one third of the total "
                     "value by their own points, and envy-free if no one values another person's share more than "
                     "their own.")
            out.append(ct.CodeTemplate(parts=(setup + " Which statement about the proposed allocation is true?",),
                                       type='envy_check', domain=dom, tags=persp, difficulty_level=1,
                                       spec={'type': 'envy_check'}))
    return out


def _names(r, k):
    return [NAMES[int(i)] for i in r.choice(len(NAMES), size=k, replace=False)]


def _join(xs):
    return ', '.join(xs[:-1]) + ' and ' + xs[-1]


def _pick(r, key: str, cands: list, k: int = 3):
    out = []
    for i in r.permutation(len(cands)):
        c = cands[int(i)]
        if c != key and c not in out:
            out.append(c)
        if len(out) == k:
            return out
    return None


# ----------------------------------------------------------------------------------------- divide and choose
def value_left(t, a, v1, v2):
    """Value of [0, t] with v1 per unit up to a and v2 after."""
    return v1 * min(t, a) + v2 * max(F(0), t - a)


def half_point(a, L, v1, v2):
    half = value_left(F(L), a, v1, v2) / 2
    return half / v1 if v1 * a >= half else a + (half - v1 * a) / v2


def _exact(x, places=2):
    return (x * 10 ** places).denominator == 1


def _divider_chooser(r, g, ask):
    L = g['length']
    while True:
        a = int(r.integers(2, 11)) * L // 12
        dv = [int(r.integers(1, 10)) for _ in range(2)]
        cv = [int(r.integers(1, 10)) for _ in range(2)]
        if dv[0] == dv[1] or cv[0] == cv[1] or dv[0] * cv[1] == dv[1] * cv[0]:
            continue
        t = half_point(F(a), L, *dv)
        if not _exact(t) or abs(t - a) < F(L, 24) or abs(t - F(L, 2)) < F(L, 24):
            continue
        tc = half_point(F(a), L, *cv)
        if not _exact(tc) or abs(tc - t) < F(L, 24):
            continue
        c_left = value_left(t, a, *cv)
        c_total = value_left(F(L), a, *cv)
        c_right = c_total - c_left
        d_half = value_left(F(L), a, *dv) / 2
        amounts = [c_left, c_right, d_half, c_total / 2, value_left(t, a, *dv)]
        if not all(_exact(x) for x in amounts) or abs(c_left - c_right) < 2:
            continue
        break
    div, cho = _names(r, 2)
    u = g['units']
    k1, k2 = g['kinds'][:2]
    values = (f"{div} values {k1} at ${dv[0]} per {g['unit']} and {k2} at ${dv[1]} per {g['unit']}; {cho} values "
              f"{k1} at ${cv[0]} per {g['unit']} and {k2} at ${cv[1]} per {g['unit']}.")
    out = {'who': f'{div} and {cho}', 'a': a, 'values': values, 'divider': div, 'chooser': cho}
    left = c_left > c_right

    def piece(is_left, worth):
        side = f'the left piece (0 to {num(t)} {u})' if is_left else f'the right piece ({num(t)} to {L} {u})'
        return f'{side}, worth ${num(worth)}'
    if ask == 'cut':
        key = f'{num(t)} {u}'
        pool = [f'{num(x)} {u}' for x in (F(L, 2), tc, F(a), half_point(F(a), L, dv[1], dv[0]),
                                           F(L) - t) if 0 < x < L]
    elif ask == 'chooser':
        key = piece(left, max(c_left, c_right))
        pool = [piece(not left, min(c_left, c_right)), piece(left, d_half), piece(not left, max(c_left, c_right)),
                piece(left, c_total / 2), piece(not left, d_half)]
    else:
        key = f'${num(d_half)}'
        pool = [f'${num(x)}' for x in (min(c_left, c_right), max(c_left, c_right), 2 * d_half, c_total / 2,
                                       d_half / 2)]
    d = _pick(r, key, pool)
    if d is None:
        return None
    out['options_list'] = [[key] + d]
    out['correct_answer'] = [[1, 0, 0, 0]]
    return out


# ----------------------------------------------------------------------------------------- last diminisher
def point_for(start, target, sections, pts):
    """The position x >= start where [start, x] is worth `target` by the section points (spread evenly)."""
    pos, need = F(0), F(target)
    for (length, p) in zip(sections, pts):
        lo, hi = pos, pos + length
        pos = hi
        if hi <= start:
            continue
        s = max(lo, start)
        avail = F(p) * (hi - s) / length
        if avail >= need and p > 0:
            return s + need * length / p
        need -= avail
    return None


def _worth_from(start, sections, pts):
    """Value of [start, end of the cake] by the section points (spread evenly)."""
    pos, total = F(0), F(0)
    for length, p in zip(sections, pts):
        lo, hi = pos, pos + length
        pos = hi
        if hi > start:
            total += F(p) * (hi - max(lo, start)) / length
    return total


def last_diminisher(order, sections, pts, n):
    """[(person, start, end)] in the order pieces are taken."""
    left, start, pieces = list(order), F(0), []
    while len(left) > 1:
        k = len(left)
        # 1/k of what the rest of the cake [start, end] is worth to p
        rest = {p: _worth_from(start, sections, pts[p]) for p in left}
        xs = {p: point_for(start, rest[p] / k, sections, pts[p]) for p in left}
        cut = xs[left[0]]
        taker = left[0]
        for p in left[1:]:
            if xs[p] < cut:
                cut, taker = xs[p], p
        pieces.append((taker, start, cut, xs))
        left.remove(taker)
        start = cut
    pieces.append((left[0], start, F(sum(sections)), {}))
    return pieces


def _last_diminisher(r, ask, n):
    while True:
        cuts = sorted(int(x) for x in r.choice(range(2, 11), size=2, replace=False))
        sections = [F(cuts[0]), F(cuts[1] - cuts[0]), F(12 - cuts[1])]
        names = _names(r, n)
        pts = {}
        total = 20 * n
        for p in names:
            while True:
                a, b = sorted(int(x) for x in r.choice(range(1, total), size=2, replace=False))
                v = [a, b - a, total - b]
                if min(v) >= 2:
                    break
            pts[p] = v
        pieces = last_diminisher(names, sections, pts, n)
        ok = True
        for _, _, _, xs in pieces[:-1]:
            vals = sorted(xs.values())
            if any(y - x < F(1, 10) for x, y in zip(vals, vals[1:])):
                ok = False
        if not ok or pieces[0][0] == names[0] and ask == 'first' and r.random() < 0.6:
            continue  # the first cutter keeps the first piece less often
        break
    flav = ['chocolate', 'vanilla', 'lemon']
    sec_text = (f"{flav[0]} from 0 to {cuts[0]} inches, {flav[1]} from {cuts[0]} to {cuts[1]} inches and {flav[2]} "
                f"from {cuts[1]} to 12 inches")
    lines = '\n'.join(f"{p}: {flav[0]} {v[0]}, {flav[1]} {v[1]}, {flav[2]} {v[2]}" for p, v in pts.items())
    out = {'who': _join(names), 'sections': sec_text, 'points': lines, 'order': _join(names)}
    if ask == 'first':
        taker, s, e, xs = pieces[0]
        key = f'{taker} gets the piece from 0 to {num(e)} inches'
        pool = [f'{p} gets the piece from 0 to {num(x)} inches' for p, x in xs.items()] + \
               [f'{p} gets the piece from 0 to {num(e)} inches' for p in names]
    else:
        def alloc(ps):
            return '; '.join(f'{p}: {num(s)} to {num(e)} inches' for p, (s, e) in ps)
        true = [(p, (s, e)) for p, s, e, _ in pieces]
        key = alloc(true)
        spans = [se for _, se in true]
        pool = [alloc(list(zip(perm, spans))) for perm in permutations([p for p, _ in true])]
        # the second piece cut at the other remaining person's point
        p2, s2, e2, xs2 = pieces[1]
        other = next(p for p in xs2 if p != p2)
        alt = [true[0], (other, (s2, xs2[other])), (p2, (xs2[other], F(12)))]
        pool.append(alloc(alt))
    d = _pick(r, key, pool)
    if d is None:
        return None
    out['options_list'] = [[key] + d]
    out['correct_answer'] = [[1, 0, 0, 0]]
    return out


# ----------------------------------------------------------------------------------------- adjusted winner
def adjusted_winner(items, va, vb, detail: bool = False):
    """(holder of the split item before the split, split item, fraction the holder keeps) and the initial holder."""
    own = {i: ('A' if va[i] > vb[i] else 'B') for i in items}
    sa = sum(va[i] for i in items if own[i] == 'A')
    sb = sum(vb[i] for i in items if own[i] == 'B')
    if sa == sb:
        return None
    lead, (vl, vt) = ('A', (va, vb)) if sa > sb else ('B', (vb, va))
    hi, lo = max(sa, sb), min(sa, sb)
    for i in sorted((i for i in items if own[i] == lead), key=lambda i: F(vl[i], vt[i])):
        if hi - vl[i] > lo + vt[i]:
            hi, lo = hi - vl[i], lo + vt[i]
            continue
        if hi - vl[i] == lo + vt[i]:
            return None
        f = F(hi - lo, vl[i] + vt[i])  # share handed over
        if detail:
            return lead, i, 1 - f, hi, lo
        return lead, i, 1 - f
    return None


def _adjusted_winner(r, dom):
    while True:
        k = int(r.integers(4, 7))
        items = [ITEMS[dom][int(i)] for i in r.choice(len(ITEMS[dom]), size=k, replace=False)]

        def spread():
            cuts = sorted(int(x) for x in r.choice(range(1, 100), size=k - 1, replace=False))
            return [b - a for a, b in zip([0] + cuts, cuts + [100])]
        va, vb = dict(zip(items, spread())), dict(zip(items, spread()))
        if any(va[i] == vb[i] for i in items) or len({F(va[i], vb[i]) for i in items}) < k:
            continue
        got = adjusted_winner(items, va, vb)
        if got is None:
            continue
        lead, split, keep = got
        if not (F(5, 100) < keep < F(95, 100)) or abs(keep - F(1, 2)) < F(2, 100):
            continue
        if (1000 * keep).denominator == 2:
            continue   # exactly on a rounding tie at one decimal (13.75%): the printed shares would depend on the rule
        break
    a, b = _names(r, 2)
    holder, other = (a, b) if lead == 'A' else (b, a)
    pts = '\n'.join(f"{it[4:].capitalize() if it.startswith('the ') else it}: {a} {va[it]}, {b} {vb[it]}"
                    for it in items)

    def opt(item, keep_):
        return (f"{item[0].upper() + item[1:]} is split: {holder} keeps {num(100 * keep_, 1)}% of it and {other} "
                f"gets {num(100 - 100 * keep_, 1)}%")
    key = opt(split, keep)
    vl, vt = (va, vb) if lead == 'A' else (vb, va)
    _, _, _, hi, lo = adjusted_winner(items, va, vb, detail=True)
    # Distractors are mistakes with their own shares (re-review, October 2026: every option repeated the key's
    # share, so only the item was tested): the right item with the shares swapped, split in half, or equalised with
    # the leader's points only ((hi - lo) / (2 * leader's points)); another item the leader's held, split by the same formula as the key.
    def ok(x):
        return F(5, 100) < x < F(95, 100)
    same = [k_ for k_ in (1 - keep, F(1, 2), 1 - F(hi - lo, 2 * vl[split])) if ok(k_)]
    wrong = [(it, 1 - F(hi - lo, vl[it] + vt[it])) for it in items if it != split and vl[it] > vt[it]]
    wrong = [(it, k_) for it, k_ in wrong if ok(k_)]
    while True:
        d, used = [], {num(100 * keep, 1)}
        cands = [(split, k_) for k_ in same]
        r.shuffle(cands)
        o2 = list(wrong)
        r.shuffle(o2)
        picks = cands[:int(r.integers(1, 3))] + o2[:2] + cands[2:] + o2[2:]
        for it, k_ in picks:
            if num(100 * k_, 1) not in used and len(d) < 3:
                d.append(opt(it, k_))
                used.add(num(100 * k_, 1))
        if len(d) == 3:
            break
        wrong.append((items[int(r.integers(len(items)))], F(int(r.integers(10, 90)), 100)))
    return {'who': f'{a} and {b}', 'points': pts, 'options_list': [[key] + d], 'correct_answer': [[1, 0, 0, 0]]}


# ----------------------------------------------------------------------------------------- envy check
def classify(alloc, vals, people):
    share = {p: sum(vals[p][i] for i in alloc[p]) for p in people}
    prop = all(3 * share[p] >= 100 for p in people)
    ef = all(share[p] >= sum(vals[p][i] for i in alloc[q]) for p in people for q in people)
    return prop, ef


def _envy_check(r, dom):
    want = int(r.integers(3))  # 0 both, 1 proportional only, 2 neither
    while True:
        k = int(r.integers(4, 6))
        items = [ITEMS[dom][int(i)] for i in r.choice(len(ITEMS[dom]), size=k, replace=False)]
        people = _names(r, 3)
        vals = {}
        for p in people:
            cuts = sorted(int(x) for x in r.choice(range(1, 100), size=k - 1, replace=False))
            vals[p] = dict(zip(items, [b - a for a, b in zip([0] + cuts, cuts + [100])]))
        owner = [int(x) for x in r.integers(0, 3, size=k)]
        if len(set(owner)) < 3:
            continue
        alloc = {p: [it for it, o in zip(items, owner) if o == j] for j, p in enumerate(people)}
        prop, ef = classify(alloc, vals, people)
        got = 0 if ef else 1 if prop else 2
        if got == want:
            break
    pts = '\n'.join(f"{p}: " + ', '.join(f'{it[4:]} {vals[p][it]}' for it in items) for p in people)
    allocation = '; '.join(f"{p} gets {_join(alloc[p]) if len(alloc[p]) > 1 else alloc[p][0]}" for p in people)
    key = EF_OPTIONS[want]
    return {'who': _join(people), 'points': pts, 'allocation': allocation, 'options_list': [EF_OPTIONS],
            'correct_answer': [[int(o == key) for o in EF_OPTIONS]]}


@function_registry.register(ELEMENT)
def fill_fair_division(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    r = filler_utils.rng
    t = spec['type']
    while True:
        if t == 'divider_chooser':
            out = _divider_chooser(r, LINEAR[spec['g']], spec['ask'])
        elif t == 'last_diminisher':
            out = _last_diminisher(r, spec['ask'], spec['n'])
        elif t == 'adjusted_winner':
            out = _adjusted_winner(r, metadata['domain'])
        else:
            out = _envy_check(r, metadata['domain'])
        if out is not None:
            return out


ct.register(ELEMENT, _templates())

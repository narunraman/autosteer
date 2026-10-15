"""sc_nondictatorship (STEER 4.1.d, Non-Dictatorial Social Welfare Function): hand-written templates.

Design of 6 October 2026 (replaces the held sc_dictatorship). A group's three-member committee ranks the shortlisted
options; every member submits a strict ranking. A rule (social welfare function) is a dictatorship if one fixed
member's ballot always becomes the committee's ranking, whatever the ballots are.

Types:
    describe (easy, difficulty 1): four rules described in words; three are dictatorships in disguise (the chair's
        ballot, the ballot of the member who registered first, the oldest member's ballot with the next one used
        only to break ties, the secretary's ballot read from top to bottom, ...) and one is not (Borda points, first
        places, or fewest last places, each with a stated tie-break). Options are the four descriptions; the key is
        the non-dictatorial one.
    consequences (medium, difficulty 2): a table of three or four meetings, each with the three members' ballots and
        the rankings that four unnamed rules produced. "Which rule cannot be a dictatorship, given these results?"
        Three rules copy one member's ballot at every meeting (a different member each); the key is a Borda,
        first-places or fewest-last-places ranking that differs from every member's ballot at some meeting while
        matching some member's ballot at all but at most one meeting.
    procedures (hard, difficulty 3): one subtle procedure in words (e.g. "if the two youngest members submit identical
        ballots, theirs is adopted; otherwise the oldest member's"); options "Yes: <member>'s ballot always becomes
        the committee's ranking" for each member and "No: no single member's ballot always does". Keys by checking
        all 216 profiles of three members over three options; half the procedures have a dictator. Procedures that
        could produce something other than a ranking are not used. Two phrasings of the question.
Members have a name, a role (chair, secretary, treasurer), an age and a registration order, all stated.
"""
from __future__ import annotations

from itertools import permutations, product

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.sc_common import GROUPS, LETTERS, NUMBER_WORDS, letters_list
from .utilities.story_utils import MEN, WOMEN

function_registry = FillRegister()

ELEMENT = 'sc_nondictatorship'
TYPES = ('describe', 'consequences', 'procedures')
PERSPECTIVES = ('anon', 'first_person', 'second_person', 'man', 'woman', 'ai_assistant')
ROLES = ('chair', 'secretary', 'treasurer')
NAMES = ['Ana', 'Ben', 'Cara', 'Dev', 'Eli', 'Femi', 'Gus', 'Hana', 'Ivo', 'Jade', 'Kofi', 'Lena', 'Milo', 'Nina',
         'Omar', 'Pia', 'Raj', 'Sara', 'Tom', 'Uma']
NO_TEXT = "No: no single member's ballot always does"


def _intro(persp: str, k: int, group: tuple) -> str:
    a_group, thing, noun = group[0], group[2], group[3]
    what = f'the rule by which its three-member committee ranks the shortlisted {noun}s when choosing {thing}'
    if persp == 'anon':
        return f'{a_group[0].upper() + a_group[1:]} is deciding on {what}.'
    if persp == 'first_person':
        return f'I am helping {a_group} decide on {what}.'
    if persp == 'second_person':
        return f'You are helping {a_group} decide on {what}.'
    if persp in ('man', 'woman'):
        name = (MEN if persp == 'man' else WOMEN)[k % 7]
        return f'{name} is helping {a_group} decide on {what}.'
    return f'You are an AI assistant helping {a_group} decide on {what}.'


QUESTIONS = {
    'describe': ("Which proposed rule is not a dictatorship, that is, under which rule is there no single member "
                 "whose ballot always becomes the committee's ranking?"),
    'consequences': ("A dictatorship is a rule under which one fixed member's ballot always becomes the committee's "
                     "ranking. Given these results, which rule cannot be a dictatorship?"),
    'procedures': ("Is there a member whose ballot always becomes the committee's ranking, whatever ballots the "
                   "members submit?"),
    'procedures_2': ("Is this rule a dictatorship, that is, does one fixed member's ballot become the committee's "
                     "ranking for every possible set of ballots?"),
}


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for g, group in enumerate(GROUPS):
        noun = group[3]
        for k, persp in enumerate(PERSPECTIVES):
            intro = _intro(persp, g + k, group)
            for qtype in TYPES:
                for m, ask in (((3, 'procedures'), (3, 'procedures_2')) if qtype == 'procedures' else
                               ((3, qtype), (4, qtype))):
                    head = (f"{intro} The {noun}s are {{alternatives}}. The committee members are {{members}}. Each "
                            f"member submits a strict ranking of all {NUMBER_WORDS[m]} {noun}s (no ties). ")
                    if qtype == 'describe':
                        body = 'Four rules are proposed (the options). '
                    elif qtype == 'consequences':
                        body = (f"Four rules were tried at {{n_meetings}} meetings. The members' ballots and the "
                                f"committee rankings produced by each rule were:\n{{table}}\n")
                    else:
                        body = 'The proposed rule: {procedure}\n'
                    out.append(ct.CodeTemplate(
                        parts=(head + body + QUESTIONS[ask],), type=qtype, domain=group[5], tags=persp,
                        difficulty_level={'describe': 1, 'consequences': 2, 'procedures': 3}[qtype],
                        spec={'group': g, 'm': m, 'type': qtype}))
    return out


# ------------------------------------------------------------------------------------------------ committee
class Committee:
    """Three members: name, role, age; registration order. Ballots are tuples, best first, keyed by name."""

    def __init__(self, r):
        self.names = [NAMES[int(i)] for i in r.choice(len(NAMES), size=3, replace=False)]
        roles = [ROLES[int(i)] for i in r.permutation(3)]
        self.role = dict(zip(self.names, roles))
        ages = sorted(int(a) for a in r.choice(range(24, 72), size=3, replace=False))
        self.age = dict(zip(self.names, [ages[int(i)] for i in r.permutation(3)]))
        self.registered = [self.names[int(i)] for i in r.permutation(3)]

    def by_role(self, role):
        return next(n for n in self.names if self.role[n] == role)

    def by_age(self):
        return sorted(self.names, key=lambda n: self.age[n])  # youngest first

    def text(self, with_registration: bool) -> str:
        parts = [f'{n} (the {self.role[n]}, age {self.age[n]})' for n in self.names]
        s = ', '.join(parts[:2]) + ' and ' + parts[2]
        if with_registration:
            s += (f'; {self.registered[0]} registered for the committee first, then {self.registered[1]}, then '
                  f'{self.registered[2]}')
        return s


def _points(ballots, cands, weights=None):
    m = len(cands)
    s = {c: 0 for c in cands}
    for name, b in ballots.items():
        w = 1 if weights is None else weights[name]
        for i, c in enumerate(b):
            s[c] += w * (m - 1 - i)
    return s


def _rank(score, cands, tie):
    """Order by score (higher first); `tie` is a sort key for equal scores."""
    return tuple(sorted(cands, key=lambda c: (-score[c], tie(c))))


# ------------------------------------------------------------------------------------- describe-type rules
def describe_rules(c: Committee, noun: str, m: int):
    """(text, function of ballots, dictator name or None) for every rule of the describe bank."""
    oldest = c.by_age()[-1]
    chair, sec = c.by_role('chair'), c.by_role('secretary')
    first_reg = c.registered[0]
    alpha = sorted(c.names)[0]
    pts = '2 points to its first choice, 1 to its second and 0 to its last' if m == 3 else \
        '3, 2, 1 and 0 points to its choices from first to last'
    dictators = [
        (f"The chair's ballot becomes the committee's ranking.", chair),
        (f"The ballot of the member who registered first becomes the committee's ranking.", first_reg),
        (f"The committee's ranking follows the oldest member's ballot; the next oldest member's ballot is used only "
         f"to break ties in it.", oldest),
        (f"The {noun} at the top of the secretary's ballot is placed first, the {noun} second on the secretary's "
         f"ballot is placed second, and so on.", sec),
        (f"The ballots are read in alphabetical order of the members' names, and the first ballot read becomes the "
         f"committee's ranking.", alpha),
    ]
    rules = [(t, (lambda d: lambda b, cands: b[d])(d), d) for t, d in dictators]

    def borda(b, cands):
        return _rank(_points(b, cands), cands, lambda x: x)

    def firsts(b, cands):
        f = {x: sum(v[0] == x for v in b.values()) for x in cands}
        s = {x: sum(v[1] == x for v in b.values()) for x in cands}
        return tuple(sorted(cands, key=lambda x: (-f[x], -s[x], x)))

    def fewest_last(b, cands):
        last = {x: sum(v[-1] == x for v in b.values()) for x in cands}
        return tuple(sorted(cands, key=lambda x: (last[x], x)))
    rules += [
        (f"Each ballot gives {pts}; the {noun}s are ranked by total points, and ties are broken alphabetically.",
         borda, None),
        (f"The {noun}s are ranked by how many members rank them first; ties are broken by the number of second "
         f"places, then alphabetically.", firsts, None),
        (f"The {noun}s are ranked by how many members rank them last, fewest first; ties are broken "
         f"alphabetically.", fewest_last, None),
    ]
    return rules


def find_dictator(rule, names, cands):
    """The member whose ballot the rule always returns (checking every profile), or None. Also checks the output is
    always a ranking of the options."""
    perms = list(permutations(cands))
    candidates = set(names)
    for prof in product(perms, repeat=len(names)):
        b = dict(zip(names, prof))
        out = rule(b, cands)
        if out is None or sorted(out) != sorted(cands) or len(out) != len(cands):
            raise ValueError('not a ranking')
        candidates = {n for n in candidates if b[n] == out}
    if len(candidates) > 1:
        raise ValueError('several dictators')
    return next(iter(candidates), None)


# ------------------------------------------------------------------------------------- procedures bank
def procedure_bank(c: Committee, noun: str):
    """(text, function) for every hard procedure; the dictator is found by `find_dictator`."""
    young, middle, old = c.by_age()
    chair, sec, tre = c.by_role('chair'), c.by_role('secretary'), c.by_role('treasurer')
    bank = []

    def two_youngest(b, cands):
        return b[young] if b[young] == b[middle] else b[old]
    bank.append(("If the two youngest members submit identical ballots, their ballot becomes the committee's ranking; "
                 "otherwise the oldest member's ballot does.", two_youngest))

    def oldest_shared(b, cands):
        for n in (young, middle):
            if b[n] == b[old]:
                return b[n]
        return b[old]
    bank.append(("If the oldest member's ballot is identical to another member's ballot, that shared ballot becomes "
                 "the committee's ranking; otherwise the oldest member's ballot does.", oldest_shared))

    for w in (2, 3):
        for tie in ('chair', 'alpha'):
            def weighted_pairs(b, cands, w=w, tie=tie):
                wins = {x: 0 for x in cands}
                for x, y in permutations(cands, 2):
                    if x < y:
                        vx = sum((w if n == chair else 1) for n in c.names if b[n].index(x) < b[n].index(y))
                        vy = sum((w if n == chair else 1) for n in c.names if b[n].index(y) < b[n].index(x))
                        if vx == vy:
                            win = (x if b[chair].index(x) < b[chair].index(y) else y) if tie == 'chair' else x
                        else:
                            win = x if vx > vy else y
                        wins[win] += 1
                if sorted(wins.values()) != list(range(len(cands))):
                    return None  # a cycle: not a ranking
                return tuple(sorted(cands, key=lambda x: -wins[x]))
            tie_text = "the chair's preference" if tie == 'chair' else f"the {noun} that comes first alphabetically"
            bank.append((f"Each pair of {noun}s is decided by a vote in which the chair has {w} votes and each other "
                         f"member one vote, and a tied vote goes to {tie_text}. The committee's ranking places each "
                         f"{noun} according to these pairwise results.", weighted_pairs))

    def chair_top_secretary(b, cands):
        top = b[chair][0]
        return (top,) + tuple(x for x in b[sec] if x != top)
    bank.append((f"The chair's first choice is placed first; the other two {noun}s follow in the order in which they "
                 f"appear on the secretary's ballot.", chair_top_secretary))

    def chair_top_vote(b, cands):
        top = b[chair][0]
        rest = [x for x in b[chair] if x != top]
        return (top, *rest)  # the chair and the secretary vote, the chair breaks ties: always the chair's order
    bank.append((f"The chair's first choice is placed first; the other two {noun}s are ordered by a vote between the "
                 f"chair and the secretary, with the chair breaking a tie.", chair_top_vote))

    for w in (3, 4, 5):
        for tie in ('treasurer', 'alpha'):
            def weighted_borda(b, cands, w=w, tie=tie):
                s = _points(b, cands, {n: (w if n == tre else 1) for n in c.names})
                key = (lambda x: b[tre].index(x)) if tie == 'treasurer' else (lambda x: x)
                return _rank(s, cands, key)
            tie_text = "by the treasurer's ballot" if tie == 'treasurer' else 'alphabetically'
            bank.append((f"Each ballot gives 2 points to its first choice, 1 to its second and 0 to its last; the "
                         f"treasurer's ballot counts {w} times and the others once. The {noun}s are ranked by total "
                         f"points, and ties are broken {tie_text}.", weighted_borda))

    def last_veto(b, cands):
        others = [n for n in c.names if n != chair]
        if b[others[0]][-1] == b[others[1]][-1]:
            x = b[others[0]][-1]
            return tuple(y for y in b[chair] if y != x) + (x,)
        return b[chair]
    bank.append((f"The chair's ballot becomes the committee's ranking, except that if the other two members rank the "
                 f"same {noun} last, that {noun} is moved to the bottom.", last_veto))

    def youngest_unless(b, cands):
        top = b[young][0]
        if all(b[n][-1] == top for n in (middle, old)):
            return b[old]
        return b[young]
    bank.append(("The youngest member's ballot becomes the committee's ranking, unless both other members rank the "
                 "youngest member's first choice last; in that case the oldest member's ballot does.", youngest_unless))

    def majority_top(b, cands):
        tops = [b[n][0] for n in c.names]
        top = next((x for x in cands if tops.count(x) >= 2), b[chair][0])
        return (top,) + tuple(x for x in b[chair] if x != top)
    bank.append((f"The {noun} ranked first by at least two members is placed first (if there is none, the chair's "
                 f"first choice); the other two follow in the chair's order.", majority_top))

    def reversed_twice(b, cands):
        rev = tuple(reversed(b[sec]))
        return tuple(reversed(rev))
    bank.append((f"The secretary's ballot is written down in reverse order, and the committee's ranking lists the "
                 f"{noun}s from the bottom of that reversed list to its top.", reversed_twice))

    def swap_if_hated(b, cands):
        others = [n for n in c.names if n != chair]
        top = b[chair][0]
        if all(b[n][-1] == top for n in others):
            return (b[chair][2], b[chair][1], b[chair][0])
        return b[chair]
    bank.append(("The chair's ballot becomes the committee's ranking, unless the other two members both rank the "
                 "chair's first choice last; then the chair's first and last choices swap places.", swap_if_hated))
    return bank


# ------------------------------------------------------------------------------------------------ filler
def _ballot(b):
    return ' > '.join(b)


def _consequences(r, c: Committee, cands, noun):
    m = len(cands)
    perms = list(permutations(cands))
    rules = describe_rules(c, noun, m)
    nondict = [f for _, f, d in rules if d is None]
    for _ in range(5000):
        k = int(r.integers(3, 5))
        meetings = [{n: perms[int(r.integers(len(perms)))] for n in c.names} for _ in range(k)]
        f = nondict[int(r.integers(len(nondict)))]
        outs = [f(b, cands) for b in meetings]
        if any(all(b[n] == o for b, o in zip(meetings, outs)) for n in c.names):
            continue  # the key would look like a dictatorship
        matched = sum(any(b[n] == o for n in c.names) for b, o in zip(meetings, outs))
        if matched < k - 1:
            continue
        return meetings, outs
    raise RuntimeError(f'{ELEMENT}: no consequences table')


@function_registry.register(ELEMENT)
def fill_sc_nondictatorship(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    group = GROUPS[spec['group']]
    noun, m, qtype = group[3], spec['m'], spec['type']
    r = filler_utils.rng
    cands = [LETTERS[i] for i in sorted(r.choice(len(LETTERS), size=m, replace=False))]
    c = Committee(r)
    if qtype == 'describe':
        # three dictatorial distractors with three different dictators (review, October 2026: two or three of
        # them often picked the same member); redraw the committee until the rule bank covers all three members
        while len({d for _, _, d in describe_rules(c, noun, m) if d is not None}) < 3:
            c = Committee(r)
    out = {'alternatives': letters_list(cands)}
    if qtype == 'describe':
        rules = describe_rules(c, noun, m)
        dict_rules = [x for x in rules if x[2] is not None]
        nd = [x for x in rules if x[2] is None]
        key = nd[int(r.integers(len(nd)))]
        others, seen = [], set()
        for i in r.permutation(len(dict_rules)):
            if dict_rules[int(i)][2] not in seen:
                others.append(dict_rules[int(i)])
                seen.add(dict_rules[int(i)][2])
            if len(others) == 3:
                break
        uses_registration = any('registered' in t for t, _, _ in others)
        chosen = [key] + others
        order = [chosen[int(i)] for i in r.permutation(4)]
        options = [t for t, _, _ in order]
        correct = [int(x is key) for x in order]
        out['members'] = c.text(uses_registration)
    elif qtype == 'consequences':
        meetings, outs = _consequences(r, c, cands, noun)
        dictators = list(c.names)
        rule_outs = [outs] + [[b[d] for b in meetings] for d in dictators]
        order = [int(i) for i in r.permutation(4)]
        lines = []
        for j, b in enumerate(meetings):
            votes = ', '.join(f'{n} voted {_ballot(b[n])}' for n in c.names)
            res = ', '.join(f'Rule {i + 1} gave {_ballot(rule_outs[order[i]][j])}' for i in range(4))
            lines.append(f'Meeting {j + 1}: {votes}. {res}.')
        out['table'] = '\n'.join(lines)
        out['n_meetings'] = NUMBER_WORDS.get(len(meetings), str(len(meetings)))
        options = [f'Rule {i + 1}' for i in range(4)]
        correct = [int(order[i] == 0) for i in range(4)]
        out['members'] = c.text(False)
    else:
        bank = procedure_bank(c, noun)
        want_yes = bool(r.random() < 0.5)
        while True:
            text, f = bank[int(r.integers(len(bank)))]
            try:
                d = find_dictator(f, c.names, cands)
            except ValueError:
                continue  # not always a ranking: not used
            if (d is not None) == want_yes:
                break
        out['procedure'] = text
        options = [f"Yes: {n}'s ballot always becomes the committee's ranking" for n in c.names] + [NO_TEXT]
        correct = [int(n == d) for n in c.names] + [int(d is None)]
        out['members'] = c.text(False)
    out['options_list'] = [options]
    out['correct_answer'] = [correct]
    return out


ct.register(ELEMENT, _templates())

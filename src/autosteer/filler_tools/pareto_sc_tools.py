"""pareto_sc (STEER 4.1.a, Pareto Efficiency in Social Welfare Functions): hand-written templates.

Decision of 3 October 2026. A group ranks two or three alternatives; the profile is written as ballot lines
"12 voters voted for X > Y > Z" (full rankings only). Two question types:
  yes_no: "The club's final decision was Book X. Does this decision satisfy the Pareto efficiency axiom?" The key is
      "No" exactly when every member ranks some other alternative above the chosen one. Profiles are drawn so that
      this is the same as Pareto dominance (one alternative that every member ranks above the chosen one); with
      three alternatives, profiles where the two readings differ are redrawn. "Yes" and "No" are equally likely.
  which: every member ranks one alternative above the other(s) (a unanimous preference, two or three
      alternatives); "According to the Pareto efficiency axiom, which book should the club choose?" The key is that
      alternative, the only Pareto-efficient one.
Candidate letters avoid A and I (they read as words). The independent audit solver
(tests/solvers/steer/solvers/voting.py, solve_pareto) must agree with every key (tests/test_pareto_sc.py).
"""
from __future__ import annotations

from itertools import permutations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

LETTERS = 'BCDEFGHJKLMNPQRSTUVWXYZ'
# (intro subject, short name, thing chosen, option noun, domain)
GROUPS = [
    ('A book club', 'club', 'its next book', 'book', 'Book', 'leisure'),
    ('A city council', 'council', 'a design for the new park', 'design', 'Design', 'politics'),
    ('A band', 'band', 'the opening song for its tour', 'song', 'Song', 'entertainment'),
    ('A family', 'family', 'a holiday destination', 'destination', 'Destination', 'family'),
    ('An office team', 'team', 'a venue for its end-of-year lunch', 'venue', 'Venue', 'workplace'),
    ('A school board', 'board', 'a new logo for the school', 'logo', 'Logo', 'education'),
]


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, (intro, short, thing, noun, Noun, domain) in enumerate(GROUPS):
        for m in (2, 3):
            head = (f"{intro} is choosing {thing} from {'two' if m == 2 else 'three'} {noun}s: {{alternatives}}. "
                    f"Each member ranked the {noun}s, with these results:\n{{ballots}}\n")
            out.append(ct.CodeTemplate(
                parts=(head + f"The {short}'s final decision was {Noun} {{chosen}}. Does this decision satisfy the "
                              f"Pareto efficiency axiom?",),
                type='yes_no', domain=domain, tags='explicit', difficulty_level=4,
                spec={'group': k, 'm': m, 'kind': 'yes_no'}))
            out.append(ct.CodeTemplate(
                parts=(head + f'According to the Pareto efficiency axiom, which {noun} should the {short} choose?',),
                type='which', domain=domain, tags='explicit', difficulty_level=4,
                spec={'group': k, 'm': m, 'kind': 'which'}))
    return out


def dominated_by(profile, x):
    """Alternatives that every voter ranks above x (profile: list of (count, ranking))."""
    alts = profile[0][1]
    return [y for y in alts if y != x and all(r.index(y) < r.index(x) for _, r in profile)]


def everyone_prefers_something(profile, x) -> bool:
    return all(r.index(x) > 0 for _, r in profile)


def _profile(r, letters, lines):
    rankings = list(permutations(letters))
    idx = r.choice(len(rankings), size=lines, replace=False)
    return [(int(r.integers(1, 41)), tuple(rankings[i])) for i in sorted(idx)]


def _fmt_alternatives(Noun, letters):
    names = [f'{Noun} {x}' for x in letters]
    return ' and '.join(names) if len(names) == 2 else f'{names[0]}, {names[1]} and {names[2]}'


@function_registry.register('pareto_sc')
def fill_pareto_sc(metadata: dict) -> dict:
    spec = ct.spec_for('pareto_sc', metadata['question_text'])
    intro, short, thing, noun, Noun, domain = GROUPS[spec['group']]
    r = filler_utils.rng
    m = spec['m']
    letters = [LETTERS[i] for i in sorted(r.choice(len(LETTERS), size=m, replace=False))]
    if spec['kind'] == 'yes_no':
        want_no = bool(r.integers(0, 2))
        while True:
            profile = _profile(r, letters, int(r.integers(1, min(4, 2 if m == 2 else 6) + 1)))
            chosen = letters[int(r.integers(0, m))]
            dom = dominated_by(profile, chosen)
            if bool(dom) != everyone_prefers_something(profile, chosen):
                continue  # the two readings of "No" differ: redraw
            if bool(dom) == want_no:
                break
        options = ['Yes', 'No']
        key = 'No' if want_no else 'Yes'
        extra = {'chosen': chosen}
    else:
        top = letters[int(r.integers(0, m))]
        rest = [x for x in letters if x != top]
        lines = 1 if m == 2 else int(r.integers(1, 3))
        orders = [tuple([top] + list(p)) for p in permutations(rest)]
        idx = sorted(r.choice(len(orders), size=lines, replace=False))
        profile = [(int(r.integers(1, 41)), orders[i]) for i in idx]
        options = [f'{Noun} {x}' for x in letters]
        key = f'{Noun} {top}'
        extra = {}
    ballots = '\n'.join(f"{n} voters voted for {' > '.join(rk)}" for n, rk in profile)
    return {'alternatives': _fmt_alternatives(Noun, letters), 'ballots': ballots, **extra,
            'options_list': [options], 'correct_answer': [[int(o == key) for o in options]]}


ct.register('pareto_sc', _templates())

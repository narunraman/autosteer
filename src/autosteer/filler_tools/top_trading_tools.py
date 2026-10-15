"""top_trading (STEER 4.4.a): run the Top Trading Cycles algorithm on a small housing-market style swap.

Generator of October 2026 (the released questions were written one by one by GPT-4, with no slots and no computed
answers). Three to five people each own one item (household items, books, sports gear, collectibles, instruments)
and rank all the items, their own included, best first: "Ana owns the lamp and ranks the items: clock > vase > lamp".
The allocation is computed here by Top Trading Cycles (`ttc`): each remaining person points to the owner of their
best remaining item, every cycle trades (each member gets the item they point to) and leaves, repeat.

Question types (the type), four options each:
    full_allocation   who ends up with which item. Distractors: the allocation of serial dictatorship in the listed
                      order and in reverse order, the first round's trades with everyone else keeping their item,
                      then random permutations; all distinct from the TTC outcome.
    first_cycle       which trading cycle forms in the first round. Profiles are drawn so that the first round has
                      exactly one cycle and it has at least two people; distractors are other cycles over the same
                      people (reversed, from later rounds, random), never one that forms in the first round.
    check_proposed    a proposed final allocation (the TTC outcome in half of the questions); options "Yes, the
                      proposal is the outcome of Top Trading Cycles." and three "No, the outcome is: ..." options, one
                      of them the TTC outcome when the proposal is wrong.
    own_item          which item one person ends up with; options four of the items.
Every draw has at least one trade. Perspectives (tags): none (an anonymous swap), ai_assistant (the reader runs the
swap) and, for own_item, second_person (the reader is one of the traders: "You own the lamp and rank ...").
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

ELEMENT = 'top_trading'
DOMAINS = {
    'household': ['lamp', 'clock', 'vase', 'rug', 'mirror', 'kettle', 'toaster', 'blender'],
    'books': ['atlas', 'novel', 'cookbook', 'dictionary', 'biography', 'almanac', 'anthology', 'thesaurus'],
    'sports': ['bicycle', 'kayak', 'skateboard', 'racket', 'helmet', 'snowboard', 'surfboard', 'tent'],
    'collectibles': ['stamp', 'coin', 'postcard', 'badge', 'poster', 'medal', 'banknote', 'figurine'],
    'instruments': ['guitar', 'violin', 'flute', 'drum', 'cello', 'trumpet', 'ukulele', 'harmonica'],
}
NAMES = ['Ana', 'Ben', 'Cara', 'Dev', 'Eli', 'Femi', 'Gus', 'Hana', 'Ivo', 'Jade', 'Kofi', 'Lena', 'Milo', 'Nina',
         'Omar', 'Pia', 'Raj', 'Sara', 'Tom', 'Uma']
TYPES = ['full_allocation', 'first_cycle', 'check_proposed', 'own_item']
INTRO = {
    '': 'A few neighbours swap {noun} using the Top Trading Cycles algorithm. Each owns one item and ranks all the '
        'items, best first:',
    'ai_assistant': 'You are an AI assistant running a swap of {noun} with the Top Trading Cycles algorithm. Each '
                    'participant owns one item and ranks all the items, best first:',
    'second_person': 'You and a few neighbours swap {noun} using the Top Trading Cycles algorithm. Each of you owns '
                     'one item and ranks all the items, best first:',
}
NOUNS = {'household': 'household items', 'books': 'books', 'sports': 'sports gear', 'collectibles': 'collectibles',
         'instruments': 'musical instruments'}
QUESTIONS = {
    'full_allocation': 'After running Top Trading Cycles to the end, who ends up with which item?',
    'first_cycle': 'Which trading cycle forms in the first round of Top Trading Cycles?',
    'check_proposed': 'Someone proposes this final allocation: {proposal}. Is it the outcome of Top Trading Cycles?',
    'own_item': 'Which item does {who} end up with after running Top Trading Cycles?',
    'own_item_you': 'Which item do you end up with after running Top Trading Cycles?',
}
YES = 'Yes, the proposal is the outcome of Top Trading Cycles.'


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for domain in DOMAINS:
        for t in TYPES:
            for persp in ('', 'ai_assistant') + (('second_person',) if t == 'own_item' else ()):
                q = QUESTIONS['own_item_you' if persp == 'second_person' else t]
                text = INTRO[persp].replace('{noun}', NOUNS[domain]) + '\n{preferences}\n' + q
                out.append(ct.CodeTemplate(parts=(text,), type=t, domain=domain, tags=persp,
                                           difficulty_level={'own_item': 1, 'first_cycle': 1}.get(t, 2),
                                           spec={'domain': domain, 'type': t, 'perspective': persp}))
    return out


def pointers(prefs: list[list[int]], remaining: set[int]) -> dict[int, int]:
    """Person -> person they point to (the owner of their best remaining item; person i owns item i)."""
    return {a: next(i for i in prefs[a] if i in remaining) for a in remaining}


def cycles(point: dict[int, int]) -> list[list[int]]:
    found, done = [], set()
    for a in sorted(point):
        path, x = [], a
        while x not in done and x not in path:
            path.append(x)
            x = point[x]
        if x in path:
            found.append(path[path.index(x):])
        done.update(path)
    return found


def ttc(prefs: list[list[int]]) -> tuple[list[int], list[list[list[int]]]]:
    """(allocation: person -> item, cycles of each round)."""
    remaining, alloc, rounds = set(range(len(prefs))), [None] * len(prefs), []
    while remaining:
        point = pointers(prefs, remaining)
        cs = cycles(point)
        rounds.append(cs)
        for c in cs:
            for a in c:
                alloc[a] = point[a]
            remaining -= set(c)
    return alloc, rounds


def serial_dictatorship(prefs, order) -> list[int]:
    taken, alloc = set(), [None] * len(prefs)
    for a in order:
        alloc[a] = next(i for i in prefs[a] if i not in taken)
        taken.add(alloc[a])
    return alloc


def draw_profile(r, n: int, need_single_first_cycle: bool):
    while True:
        prefs = [[int(x) for x in r.permutation(n)] for _ in range(n)]
        alloc, rounds = ttc(prefs)
        if alloc == list(range(n)):
            continue  # nobody trades
        if need_single_first_cycle and (len(rounds[0]) != 1 or len(rounds[0][0]) < 2):
            continue
        return prefs, alloc, rounds


@function_registry.register(ELEMENT)
def fill_top_trading(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    t, persp = spec['type'], spec['perspective']
    r = filler_utils.rng
    n = int(r.integers(3, 6))
    items = [DOMAINS[spec['domain']][i] for i in r.choice(len(DOMAINS[spec['domain']]), size=n, replace=False)]
    names = [NAMES[i] for i in r.choice(len(NAMES), size=n, replace=False)]
    you = persp == 'second_person'
    if you:
        names[0] = 'You'
    prefs, alloc, rounds = draw_profile(r, n, t == 'first_cycle')
    lines = []
    for a in range(n):
        verb = ('own', 'rank') if names[a] == 'You' else ('owns', 'ranks')
        lines.append(f"{names[a]} {verb[0]} the {items[a]} and {verb[1]} the items: "
                     + ' > '.join(items[i] for i in prefs[a]))
    out = {'preferences': '\n'.join(lines)}

    def gets(a, i):
        return f"{names[a]} {'get' if names[a] == 'You' else 'gets'} the {items[i]}"

    def alloc_text(al):
        return ', '.join(gets(a, al[a]) for a in range(n))

    def distinct_allocs(first, k):
        """k allocations different from `first` and from each other: plausible mistakes first, then random."""
        point = pointers(prefs, set(range(n)))
        first_round = list(range(n))
        for c in rounds[0]:
            for a in c:
                first_round[a] = point[a]
        got = []
        for c in (serial_dictatorship(prefs, range(n)), serial_dictatorship(prefs, range(n - 1, -1, -1)),
                  first_round):
            if c != first and c not in got:
                got.append(c)
        while len(got) < k:
            p = [int(x) for x in r.permutation(n)]
            if p != first and p not in got:
                got.append(p)
        return got[:k]

    if t == 'full_allocation':
        options = [alloc_text(alloc)] + [alloc_text(d) for d in distinct_allocs(alloc, 3)]
        key = options[0]
    elif t == 'first_cycle':
        point = pointers(prefs, set(range(n)))
        c = rounds[0][0]
        true = {a: point[a] for a in c}

        def cyc_text(to):
            return ', '.join(gets(a, to[a]) for a in sorted(to))
        cands = [{c[(k + 1) % len(c)]: c[k] for k in range(len(c))}]  # the cycle run backwards
        cands += [{a: alloc[a] for a in lc} for later in rounds[1:] for lc in later if len(lc) >= 2]
        opts = []
        for to in cands:
            if to != true and to not in opts:
                opts.append(to)
        while len(opts) < 3:
            m = int(r.integers(2, n + 1))
            cyc = [int(x) for x in r.choice(n, size=m, replace=False)]
            to = {cyc[k]: cyc[(k + 1) % m] for k in range(m)}
            if to != true and to not in opts:
                opts.append(to)
        options = [cyc_text(true)] + [cyc_text(to) for to in opts[:3]]
        key = options[0]
    elif t == 'check_proposed':
        correct = bool(r.integers(0, 2))
        wrong = distinct_allocs(alloc, 3)
        proposal = alloc if correct else wrong[0]
        nos = wrong if correct else [alloc] + wrong[1:]
        options = [YES] + [f'No, the outcome is: {alloc_text(d)}.' for d in nos]
        key = YES if correct else f'No, the outcome is: {alloc_text(alloc)}.'
        out['proposal'] = alloc_text(proposal)
    else:  # own_item
        who = 0 if you else int(r.integers(0, n))
        mine = alloc[who]
        others = [i for i in range(n) if i != mine]
        pick = [others[i] for i in r.choice(len(others), size=min(3, len(others)), replace=False)]
        options = [f'The {items[mine]}'] + [f'The {items[i]}' for i in pick]
        key = options[0]
        if not you:
            out['who'] = names[who]
    out['options_list'] = [options]
    out['correct_answer'] = [[int(o == key) for o in options]]
    return out


ct.register(ELEMENT, _templates())

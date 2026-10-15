"""top_trading: keys recomputed from the question text with the audit's independent TTC implementation.

The ownership and ranking lines ("Ana owns the lamp and ranks the items: clock > vase > lamp") are parsed from the
text; the final allocation comes from `_ttc` in tests/solvers/steer/solvers/mechanisms.py (written for the audit,
not by this generator). The first round is recomputed here: everyone points to the owner of their top item, and
the cycles of that graph are found by walking it. Checks on 1,200 seeded samples:
  full_allocation   exactly the keyed option states the TTC allocation;
  first_cycle       the first round has exactly one cycle and the keyed option states exactly its trades; no other
                    option is a first-round cycle;
  check_proposed    the key is "Yes" iff the proposal is the TTC allocation, otherwise the option stating it;
                    exactly one option states the TTC allocation or affirms a correct proposal;
  own_item          the keyed item is the asked person's TTC item;
plus distinct options, at least one trade, 3 to 5 people, all types / perspectives / domains, and seeded
reproducibility.
"""
import collections
import re

from helpers import load_steer_solver, sample
from autosteer.sampler import ElementSampler

MECH = load_steer_solver('mechanisms')
LINE = re.compile(r'^(\w+) owns? the (\w+) and ranks? the items: (\w+(?: > \w+)+)$', re.M)
GETS = re.compile(r'(\w+) gets? the (\w+)')
YES = 'Yes, the proposal is the outcome of Top Trading Cycles.'


def parse_alloc(text):
    return dict(GETS.findall(text))


def first_round_cycles(owner_of, prefs):
    item_of = {a: i for i, a in owner_of.items()}
    point = {a: owner_of[prefs[a][0]] for a in prefs}
    cycles = []
    for a in prefs:
        path, x = [], a
        while x not in path:
            path.append(x)
            x = point[x]
        cyc = path[path.index(x):]
        if set(cyc) not in [set(c) for c in cycles]:
            cycles.append(cyc)
    return [{a: item_of[point[a]] for a in c} for c in cycles]


def test_keys_by_independent_ttc():
    seen = collections.Counter()
    for s in sample('top_trading', 1200, seed=4404):
        lines = LINE.findall(s.question)
        assert 3 <= len(lines) <= 5, s.question
        owner_of = {item: a for a, item, _ in lines}
        prefs = {a: rank.split(' > ') for a, _, rank in lines}
        assert all(sorted(p) == sorted(owner_of) for p in prefs.values())
        alloc = MECH._ttc(owner_of, prefs)
        assert alloc and any(owner_of[i] != a for a, i in alloc.items()), s.question  # someone trades
        assert len(set(s.options)) == len(s.options) >= 3
        t = s.type
        if t == 'full_allocation':
            hits = [k for k, o in enumerate(s.options) if parse_alloc(o) == alloc]
            assert hits == [s.answer_index], (s.question, s.options, alloc)
            assert all(set(parse_alloc(o)) == set(prefs) for o in s.options)
        elif t == 'first_cycle':
            cycles = first_round_cycles(owner_of, prefs)
            assert len(cycles) == 1 and len(cycles[0]) >= 2, cycles
            hits = [k for k, o in enumerate(s.options) if parse_alloc(o) in cycles]
            assert hits == [s.answer_index], (s.question, s.options, cycles)
        elif t == 'check_proposed':
            prop = parse_alloc(re.search(r'proposes this final allocation: (.*?)\. Is it', s.question).group(1))
            assert set(prop) == set(prefs)
            correct = [k for k, o in enumerate(s.options)
                       if (o == YES and prop == alloc) or (o != YES and parse_alloc(o) == alloc)]
            assert correct == [s.answer_index], (s.question, s.options, alloc)
            assert YES in s.options
            seen[('check', prop == alloc)] += 1
        else:
            who = 'You' if s.metadata['tags'] == 'second_person' else \
                re.search(r'Which item does (\w+) end up with', s.question).group(1)
            assert s.answer == f'The {alloc[who]}', (s.question, s.options, alloc)
            assert all(o[4:] in owner_of for o in s.options)
        seen[t] += 1
        seen[('n', len(lines))] += 1
        seen[s.metadata['tags']] += 1
        seen[s.metadata['domain']] += 1
    for k in ['full_allocation', 'first_cycle', 'check_proposed', 'own_item', ('n', 3), ('n', 4), ('n', 5),
              ('check', True), ('check', False), '', 'ai_assistant', 'second_person', 'household', 'books', 'sports',
              'collectibles', 'instruments']:
        assert seen[k] > 20, (k, seen)


def test_seeded_reproducibility():
    s = ElementSampler()
    for seed in range(20):
        assert s.sample_question('top_trading', 5 + seed) == s.sample_question('top_trading', 5 + seed)

"""sc_nondictatorship: dictators found by simulating the described rules, and the results table checked.

Independent of the generator: the committee (names, roles, ages, registration order) is parsed from the question;
each rule or procedure description is mapped to a function written here from its wording (numbers such as the
chair's votes or the treasurer's weight parsed from the text).
  describe      on 400 random profiles (and all of them with three options), the three dictatorial options always
                return one fixed member's ballot and the keyed option differs from every member's ballot somewhere.
  consequences  the table is parsed: the keyed rule differs from each member's ballot at some meeting, and every
                other rule equals one member's ballot at every meeting.
  procedures    all 216 profiles of three members over three options: the procedure always returns a ranking; the
                key is "Yes: <member>..." iff that member's ballot is always returned, otherwise "No".
Also the Yes / No balance of procedures, all types, and seeded reproducibility.
"""
import collections
import random
import re
from itertools import permutations, product

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'sc_nondictatorship'
MEMBER = re.compile(r'(\w+) \(the (chair|secretary|treasurer), age (\d+)\)')
REG = re.compile(r'; (\w+) registered for the committee first, then (\w+), then (\w+)\.')
MEETING = re.compile(r'^Meeting \d+: (.+?)\. (Rule 1 gave .+)\.$', re.M)
VOTE = re.compile(r'(\w+) voted ([A-Z](?: > [A-Z])+)')
GAVE = re.compile(r'Rule (\d) gave ([A-Z](?: > [A-Z])+)')


class Members:
    def __init__(self, text):
        found = MEMBER.findall(text)
        assert len(found) == 3, text
        self.names = [n for n, _, _ in found]
        self.role = {role: n for n, role, _ in found}
        self.age = {n: int(a) for n, _, a in found}
        m = REG.search(text)
        self.registered = list(m.groups()) if m else None
        self.young, self.middle, self.old = sorted(self.names, key=self.age.get)


def points(b, alts, weight=lambda n: 1):
    return {a: sum(weight(n) * (len(alts) - 1 - r.index(a)) for n, r in b.items()) for a in alts}


def describe_rule(desc, mem, alts):
    if desc.startswith("The chair's ballot"):
        return lambda b: b[mem.role['chair']]
    if 'registered first' in desc:
        return lambda b: b[mem.registered[0]]
    if "oldest member's ballot" in desc:
        return lambda b: b[mem.old]
    if "secretary's ballot is placed second" in desc:
        return lambda b: b[mem.role['secretary']]
    if 'alphabetical order of the members' in desc:
        return lambda b: b[min(mem.names)]
    if 'total points' in desc:
        return lambda b: tuple(sorted(alts, key=lambda a: (-points(b, alts)[a], a)))
    if 'rank them first' in desc:
        def f(b):
            first = collections.Counter(r[0] for r in b.values())
            second = collections.Counter(r[1] for r in b.values())
            return tuple(sorted(alts, key=lambda a: (-first[a], -second[a], a)))
        return f
    assert 'rank them last, fewest first' in desc, desc
    return lambda b: tuple(sorted(alts, key=lambda a: (sum(r[-1] == a for r in b.values()), a)))


def procedure(desc, mem, alts):
    chair, sec, tre = mem.role['chair'], mem.role['secretary'], mem.role['treasurer']
    others = [n for n in mem.names if n != chair]
    if desc.startswith('If the two youngest'):
        return lambda b: b[mem.young] if b[mem.young] == b[mem.middle] else b[mem.old]
    if desc.startswith("If the oldest member's ballot is identical"):
        return lambda b: b[mem.old]  # a shared ballot is the oldest member's ballot
    m = re.search(r'the chair has (\d) votes and each other member one vote, and a tied vote goes to (.+?)\.', desc)
    if m:
        w, tie = int(m.group(1)), m.group(2)

        def f(b):
            beats = collections.Counter()
            for x, y in permutations(alts, 2):
                vx = sum((w if n == chair else 1) for n in mem.names if b[n].index(x) < b[n].index(y))
                vy = sum((w if n == chair else 1) for n in mem.names if b[n].index(y) < b[n].index(x))
                if vx > vy or (vx == vy and (b[chair].index(x) < b[chair].index(y) if 'chair' in tie else x < y)):
                    beats[x] += 1
            if sorted(beats[a] for a in alts) != list(range(len(alts))):
                return None
            return tuple(sorted(alts, key=lambda a: -beats[a]))
        return f
    if "order in which they appear on the secretary's ballot" in desc:
        return lambda b: (b[chair][0],) + tuple(a for a in b[sec] if a != b[chair][0])
    if 'vote between the chair and the secretary' in desc:
        def f(b):
            top = b[chair][0]
            x, y = [a for a in alts if a != top]
            c_x = b[chair].index(x) < b[chair].index(y)
            s_x = b[sec].index(x) < b[sec].index(y)
            first = x if c_x == s_x and c_x else y if c_x == s_x else (x if c_x else y)  # the chair breaks a tie
            return (top, first, y if first == x else x)
        return f
    m = re.search(r"treasurer's ballot counts (\d) times and the others once\. .* ties are broken (.+)\.$", desc)
    if m:
        w, tie = int(m.group(1)), m.group(2)

        def f(b):
            s = points(b, alts, lambda n: w if n == tre else 1)
            return tuple(sorted(alts, key=lambda a: (-s[a], b[tre].index(a) if 'treasurer' in tie else a)))
        return f
    if 'that venue is moved' in desc or 'is moved to the bottom' in desc:
        def f(b):
            x, y = (b[n][-1] for n in others)
            if x == y:
                return tuple(a for a in b[chair] if a != x) + (x,)
            return b[chair]
        return f
    if desc.startswith("The youngest member's ballot"):
        def f(b):
            top = b[mem.young][0]
            return b[mem.old] if b[mem.middle][-1] == top and b[mem.old][-1] == top else b[mem.young]
        return f
    if 'ranked first by at least two members' in desc:
        def f(b):
            tops = collections.Counter(r[0] for r in b.values())
            top = next((a for a, k in tops.items() if k >= 2), b[chair][0])
            return (top,) + tuple(a for a in b[chair] if a != top)
        return f
    if 'reverse order' in desc:
        return lambda b: b[sec]
    assert 'first and last choices swap places' in desc, desc

    def f(b):
        if all(b[n][-1] == b[chair][0] for n in others):
            return tuple(reversed(b[chair]))
        return b[chair]
    return f


def dictators(f, mem, profiles):
    left = set(mem.names)
    for prof in profiles:
        b = dict(zip(mem.names, prof))
        out = f(b)
        assert out is not None and sorted(out) == sorted(prof[0]), (out, b)
        left = {n for n in left if b[n] == out}
    return left


def test_keys():
    s = ElementSampler()
    seen = collections.Counter()
    rnd = random.Random(5)
    for seed in range(450):
        (row,) = build_rows(s.sample_question(E, 73000 + seed), 'test')
        text = row['question_text']
        mem = Members(text)
        alts = sorted(re.search(r'are ([A-Z](?:, [A-Z])* and [A-Z])\.', text).group(1).replace(' and ', ', ')
                      .split(', '))
        perms = list(permutations(alts))
        all_profiles = list(product(perms, repeat=3))
        profiles = all_profiles if len(alts) == 3 else [tuple(rnd.choice(perms) for _ in range(3))
                                                        for _ in range(400)]
        t = row['type']
        seen[t] += 1
        if t == 'describe':
            for i, o in enumerate(row['options']):
                d = dictators(describe_rule(o, mem, alts), mem, profiles)
                assert len(d) == (0 if i == row['correct_index'] else 1), (o, d)
        elif t == 'consequences':
            meetings = MEETING.findall(text)
            assert len(meetings) in (3, 4)
            copies = collections.defaultdict(set)
            for votes, gave in meetings:
                b = dict(VOTE.findall(votes))
                for k, ranking in GAVE.findall(gave):
                    copies[int(k)].add(frozenset(n for n in mem.names if b[n] == ranking) or frozenset())
            for k in (1, 2, 3, 4):
                always = set(mem.names)
                for match in copies[k]:
                    always &= match
                if len(copies[k]) == 0:
                    always = set()
                is_key = row['options'][row['correct_index']] == f'Rule {k}'
                assert bool(always) != is_key, (k, text)
        else:
            f = procedure(text.split('The proposed rule: ')[1].split('\n')[0], mem, alts)
            d = dictators(f, mem, all_profiles)
            assert len(d) <= 1
            key = row['options'][row['correct_index']]
            if d:
                assert key.startswith(f'Yes: {next(iter(d))}\'s'), (key, d)
                seen['yes'] += 1
            else:
                assert key.startswith('No'), key
                seen['no'] += 1
    assert seen['describe'] and seen['consequences'] and seen['procedures'], seen
    assert 0.3 <= seen['yes'] / seen['procedures'] <= 0.7, seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q

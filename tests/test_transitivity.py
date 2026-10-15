"""transitivity: the key is recomputed from the stated preferences in the text, independently of the filler.

Each preference sentence ("prefer(s) a over b", "like(s) a more than b", "like(s) b less than a", "find(s) b less
appealing than a") is parsed into an edge a > b. Checks on 1,000 seeded samples: the stated preferences are acyclic
(a topological order exists); the two options are ordered by the transitive closure but not stated directly; the key
is the preferred one for most_preferred and the other for least_preferred; three and four alternatives, all four
perspectives, five domains and both directions occur; the audit solver (tests/solvers/steer/solvers/axioms.py,
solve_transitivity_one) never disagrees with a key where it answers; the same seed gives the same question.
"""
import collections
import re

from helpers import load_steer_solver, sample
from autosteer.sampler import ElementSampler

AXIOMS = load_steer_solver('axioms')
SUBJ = r'(?:You|He|She|[A-Z][a-z]+)'
PATS = [(re.compile(rf'{SUBJ} prefers? (.+) over (.+)\.'), False),
        (re.compile(rf'{SUBJ} likes? (.+) more than (.+)\.'), False),
        (re.compile(rf'{SUBJ} likes? (.+) less than (.+)\.'), True),
        (re.compile(rf'{SUBJ} finds? (.+) less appealing than (.+)\.'), True)]


def norm(x):
    return x[0].lower() + x[1:]


def topo_order(nodes, edges):
    indeg = {n: 0 for n in nodes}
    for _, b in edges:
        indeg[b] += 1
    order, ready = [], [n for n in nodes if indeg[n] == 0]
    while ready:
        n = ready.pop()
        order.append(n)
        for a, b in edges:
            if a == n:
                indeg[b] -= 1
                if indeg[b] == 0:
                    ready.append(b)
    return order if len(order) == len(nodes) else None


def reaches(edges, a, b):
    seen, todo = set(), [a]
    while todo:
        x = todo.pop()
        for u, v in edges:
            if u == x and v not in seen:
                seen.add(v)
                todo.append(v)
    return b in seen


def test_keys_from_stated_preferences():
    got = sample('transitivity', 1000, seed=4303)
    seen = collections.Counter()
    agree = 0
    for s in got:
        edges = []
        for sent in re.split(r'(?<=\.) ', s.question):
            for pat, flip in PATS:
                m = pat.fullmatch(sent)
                if m:
                    a, b = m.groups()
                    edges.append((norm(b), norm(a)) if flip else (norm(a), norm(b)))
                    break
        nodes = {x for e in edges for x in e}
        assert len(edges) == len(nodes) - 1 and len(nodes) in (3, 4), (s.question, edges)
        assert topo_order(nodes, edges) is not None  # no cycle
        x, y = (norm(o) for o in s.options)
        assert len(s.options) == 2 and x != y and {x, y} <= nodes, (s.options, nodes)
        assert (x, y) not in edges and (y, x) not in edges
        xy, yx = reaches(edges, x, y), reaches(edges, y, x)
        assert xy != yx, (s.question, s.options)  # ordered, one way only
        better = x if xy else y
        want = better if s.type == 'most_preferred' else ({x, y} - {better}).pop()
        assert norm(s.answer) == want, (s.question, s.options, s.answer)
        seen[('n', len(nodes))] += 1
        seen[s.type] += 1
        seen[s.metadata['tags']] += 1
        seen[s.metadata['domain']] += 1
        pred, status = AXIOMS.solve_transitivity_one(s.question, list(s.options))
        if status == 'ok':
            agree += 1
            assert pred == s.answer_index, (s.question, s.options, pred)
    for k in [('n', 3), ('n', 4), 'most_preferred', 'least_preferred', 'second_person', 'third_person;man',
              'third_person;woman', 'ai_assistant', 'jobs', 'countries', 'fruits', 'people', 'activities']:
        assert seen[k] > 50, (k, seen)


def test_seeded_reproducibility():
    s = ElementSampler()
    for seed in range(20):
        assert s.sample_question('transitivity', 31 + seed) == s.sample_question('transitivity', 31 + seed)

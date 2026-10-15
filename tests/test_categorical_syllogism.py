"""categorical_syllogism: the key is recomputed from the question text by model checking, independently of the filler.

Each premise and option "All / No / Some X are (not) Y." is parsed from the text; with the three terms of the
premises, every way of occupying the 7 Venn regions with at least one individual per term is a model (existential
import, which the question states). A conclusion must be true iff it holds in every model of both premises. Checks
on 1,000 seeded samples: exactly the keyed option must be true, and "Cannot be determined." is the key iff no listed
conclusion must be true; four distinct options; both kinds of key and every conclusion form occur as the key; the
audit solver (tests/solvers/steer/solvers/logic_tom.py, solve_categorical_row) agrees on every question; the same
seed gives the same question.
"""
import collections
import itertools
import re

from helpers import load_steer_solver, sample
from autosteer.sampler import ElementSampler

LOGIC = load_steer_solver('logic_tom')
SENT = re.compile(r'^(All|No|Some) ([a-z -]+?) are (not )?([a-z -]+?)\.$')
CBD = 'Cannot be determined.'


def parse(s):
    m = SENT.match(s)
    assert m, s
    q, x, neg, y = m.groups()
    form = {'All': 'A', 'No': 'E', 'Some': 'O' if neg else 'I'}[q]
    assert not (neg and q != 'Some'), s
    return form, x, y


def true_in(form, x, y, world):
    """world: list of individuals, each the frozenset of terms it belongs to."""
    xs = [w for w in world if x in w]
    return {'A': all(y in w for w in xs), 'E': not any(y in w for w in xs),
            'I': any(y in w for w in xs), 'O': any(y not in w for w in xs)}[form]


def must_be_true(premises, conclusion, terms):
    kinds = [frozenset(c) for n in range(1, 4) for c in itertools.combinations(terms, n)]
    found = False
    for k in range(1, len(kinds) + 1):
        for world in itertools.combinations(kinds, k):
            if not all(any(t in w for w in world) for t in terms):
                continue
            if all(true_in(*p, world) for p in premises):
                found = True
                if not true_in(*conclusion, world):
                    return False
    assert found
    return True


def test_keys_by_model_checking():
    keys, forms, domains = collections.Counter(), collections.Counter(), collections.Counter()
    got = sample('categorical_syllogism', 1000, seed=4101)
    for s in got:
        sents = re.findall(r'(?:All|No|Some) [^.?]+\.', s.question)
        assert len(sents) == 2, s.question
        prem = [parse(x) for x in sents]
        terms = sorted({t for _, a, b in prem for t in (a, b)})
        assert len(terms) == 3
        assert len(s.options) == 4 and len(set(s.options)) == 4 and CBD in s.options
        truth = []
        for o in s.options:
            if o == CBD:
                truth.append(None)
                continue
            c = parse(o)
            assert {c[1], c[2]} <= set(terms) and c[1] != c[2]
            truth.append(must_be_true(prem, c, terms))
        follows = [i for i, t in enumerate(truth) if t]
        assert len(follows) <= 1, (s.question, s.options)
        want = follows[0] if follows else s.options.index(CBD)
        assert s.answer_index == want, (s.question, s.options, truth)
        keys['cbd' if not follows else 'valid'] += 1
        if follows:
            forms[parse(s.answer)[0]] += 1
        domains[s.metadata['domain']] += 1
        # the audit solver reads the two premises and the options
        pred, status = LOGIC.solve_categorical_row(
            ' '.join(sents) + ' Solely from the information provided, which of the following must be true?',
            s.options, True)
        assert status == 'ok' and pred == s.answer_index, (status, pred, s.question, s.options)
    assert min(keys.values()) > 400, keys
    assert set(forms) == set('AEIO'), forms
    assert set(domains) == {'people', 'animals_plants', 'objects', 'fictional'}


def test_seeded_reproducibility():
    s = ElementSampler()
    for seed in range(20):
        assert s.sample_question('categorical_syllogism', 777 + seed) == s.sample_question('categorical_syllogism',
                                                                                             777 + seed)

"""conditional_syllogism: keys recomputed from the question text by truth table, independently of the filler's forms.

The question's "If A, then B." sentence is located in the rule bank, which gives the four clauses P, not-P, Q, not-Q
of that rule (with the person's name from the text). The second premise and every option other than "Cannot be
determined." are mapped to literals by exact match against those clauses. An option must be true iff it holds in
every row of the truth table over (P, Q) where "if P then Q" and the second premise hold; the key must be the only
such option, or "Cannot be determined." when no option must be true. Checks on 1,000 seeded samples: that, three
distinct options, all four argument forms and both kinds of key occur, every domain occurs, the type names the
argument form (ponens / tollens are determinate, the two fallacies are not); the audit solver
(tests/solvers/steer/solvers/logic_tom.py, solve_conditional) never disagrees with a key where it gives an answer;
the same seed gives the same question.
"""
import collections
import itertools
import re

import pandas as pd

from helpers import load_steer_solver, sample
from autosteer.filler_tools.conditional_syllogism_tools import CBD, MEN, WOMEN, rules
from autosteer.sampler import ElementSampler

LOGIC = load_steer_solver('logic_tom')
RULE = re.compile(r'(?:^|(?<=\. ))(If [^.]*?, then [^.]*?\.) ')
LIT = {'P': ('P', True), 'nP': ('P', False), 'Q': ('Q', True), 'nQ': ('Q', False)}


def cap(s):
    return s[0].upper() + s[1:]


def bank_entry(question):
    rule_text = RULE.search(question).group(1)
    hits = []
    for r in rules():
        for name in (MEN + WOMEN if r['tags'] else ['']):
            if cap(r['rule'].format(name=name)) == rule_text:
                hits.append((r, name))
    assert len(hits) == 1, rule_text
    return hits[0]


def literal(sentence, r, name):
    found = [LIT[k] for k in LIT if cap(r[k].format(name=name)) + '.' == sentence]
    assert len(found) == 1, (sentence, found)
    return found[0]


def must_hold(premises, lit):
    rows = [dict(P=p, Q=q) for p, q in itertools.product((False, True), repeat=2)]
    models = [m for m in rows if ((not m['P']) or m['Q']) and all(m[v] == val for v, val in premises)]
    assert models
    return all(m[lit[0]] == lit[1] for m in models)


def test_keys_by_truth_table():
    got = sample('conditional_syllogism', 1000, seed=4202)
    seen, domains = collections.Counter(), set()
    rows = []
    for s in got:
        r, name = bank_entry(s.question)
        after_rule = s.question.split(RULE.search(s.question).group(1), 1)[1].strip()
        fact = after_rule.split('. ', 1)[0] + '.'
        prem = literal(fact, r, name)
        assert len(s.options) == 3 and len(set(s.options)) == 3 and CBD in s.options
        must = [o != CBD and must_hold([prem], literal(o, r, name)) for o in s.options]
        want = must.index(True) if any(must) else s.options.index(CBD)
        assert sum(must) <= 1 and s.answer_index == want, (s.question, s.options)
        form = {('P', True): 'modus_ponens', ('Q', False): 'modus_tollens', ('Q', True): 'affirming_consequent',
                ('P', False): 'denying_antecedent'}[prem]
        assert s.type == form
        assert (s.answer == CBD) == (form in ('affirming_consequent', 'denying_antecedent'))
        seen[form] += 1
        domains.add(s.metadata['domain'])
        intro = r['intro']
        rows.append((s.question[len(intro):].strip() if intro else s.question, s.options, s.answer_index))
    assert set(seen) == {'modus_ponens', 'modus_tollens', 'affirming_consequent', 'denying_antecedent'}
    assert min(seen.values()) > 180, seen
    assert domains == {'travel', 'work', 'school', 'health', 'finance', 'daily_life', 'sports', 'numbers'}
    df = pd.DataFrame({'question_text': [q for q, _, _ in rows], 'options': [o for _, o, _ in rows]})
    res = LOGIC.solve_conditional(df)
    ok = 0
    for (q, opts, key), (_, row) in zip(rows, res.iterrows()):
        if row['status'] == 'ok':
            ok += 1
            assert row['pred'] == key, (q, opts, row['pred'])
    assert ok >= 0.5 * len(rows), ok


def test_seeded_reproducibility():
    s = ElementSampler()
    for seed in range(20):
        assert s.sample_question('conditional_syllogism', 99 + seed) == s.sample_question('conditional_syllogism',
                                                                                            99 + seed)

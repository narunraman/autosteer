"""equiv_contrapositive (filler_tools/logic_tools.py, contexts in filler_tools/contrapositive_bank.json).

Independent of the filler:
  - truth table: the question's "if P, then Q" is located in the bank; every option "If A, then B." is mapped back to
    literals by exact match against the context's four clause strings, and is logically equivalent to the statement
    iff A -> B and P -> Q agree on all four truth assignments. Exactly the keyed option must be equivalent, on 1,000
    seeded samples;
  - style: the statement in the question is "P, then Q" with the context's own strings, and every option is built from
    exactly those clause strings (no option has wording of its own), so the key cannot be told apart by style;
  - negations: every not-X of the bank turns back into X by undoing one negation ("does not V" -> V in the third
    person, "do not V" -> V, "AUX not" -> AUX, "there is no" -> "there is"), with the same subject and tense;
  - 3 or 4 distinct options, no unfilled placeholders, uniform key position, domain mix, reproducibility, and the
    audit solver (tests/solvers/steer/solvers/logic_tom.py) never contradicts the key and scores most questions.
"""
import collections
import importlib.util
import itertools
import json
import os
import re

import pandas as pd

from helpers import sample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = json.load(open(os.path.join(ROOT, 'src', 'autosteer', 'filler_tools', 'contrapositive_bank.json')))
N = int(os.environ.get('STEER_TEST_N', 1000))
SEED = 20261002
PLACEHOLDER = re.compile(r'\{[A-Za-z_][A-Za-z_0-9]*\}')
LIT = {'P': ('P', False), 'Q': ('Q', False), 'notP': ('P', True), 'notQ': ('Q', True)}
_CACHE = []


def got():
    if not _CACHE:
        _CACHE.extend(sample('equiv_contrapositive', N, seed=SEED))
    return _CACHE


def context_for(question: str):
    hits = [c for c in BANK if question.endswith(f"{c['if_word']} {c['P']}, then {c['Q']}{c['outro']}")]
    assert len(hits) >= 1, question
    # contexts with the same statement and outro differ only in the intro; all have the same clauses
    assert len({(c['P'], c['Q'], c['notP'], c['notQ']) for c in hits}) == 1, question
    return hits[0]


def literal(clause: str, c):
    found = {LIT[k] for k in LIT if c[k] == clause}
    assert len(found) == 1, (clause, found)
    return found.pop()


def holds(lit, p, q):
    v = p if lit[0] == 'P' else q
    return (not v) if lit[1] else v


def equivalent(a, b) -> bool:
    return all(((not holds(a, p, q)) or holds(b, p, q)) == ((not p) or q)
               for p, q in itertools.product((False, True), repeat=2))


def test_key_by_truth_table():
    for s in got():
        c = context_for(s.question)
        eq = []
        for o in s.options:
            m = re.fullmatch(r'If (.+), then (.+)\.', o)
            assert m, o
            eq.append(equivalent(literal(m.group(1), c), literal(m.group(2), c)))
        assert sum(eq) == 1 and eq.index(True) == s.answer_index, (s.question, s.options, eq)


def test_every_option_from_the_same_clause_strings():
    for s in got():
        c = context_for(s.question)
        clauses = [c['P'], c['Q'], c['notP'], c['notQ']]
        allowed = {f'If {a}, then {b}.' for a in clauses for b in clauses if a != b}
        assert set(s.options) <= allowed, (s.question, s.options)
        assert f"{c['P']}, then {c['Q']}" in s.question


def _inflect3(lemma):
    special = {'have': 'has', 'do': 'does', 'go': 'goes', 'be': 'is'}
    if lemma in special:
        return special[lemma]
    if re.search(r'(s|sh|ch|x|z|o)$', lemma):
        return lemma + 'es'
    if re.search(r'[^aeiou]y$', lemma):
        return lemma[:-1] + 'ies'
    return lemma + 's'


def positive(neg: str) -> set[str]:
    """The clauses that neg negates (one negation undone)."""
    out = set()
    m = re.search(r'\bdoes not (\w+)', neg)
    if m:
        out.add(neg[:m.start()] + _inflect3(m.group(1)) + neg[m.end():])
    m = re.search(r'\bdo not (\w+)', neg)
    if m:
        out.add(neg[:m.start()] + m.group(1) + neg[m.end():])
    m = re.search(r'\b(is|are|am|was|were|will|has|have) not\b', neg)
    if m:
        out.add(neg[:m.start()] + m.group(1) + neg[m.end():])
    m = re.search(r'\bthere (is|was) no\b', neg)
    if m:
        out.add(neg[:m.start()] + f'there {m.group(1)}' + neg[m.end():])
        out.add(neg[:m.start()] + f'there {m.group(1)} a' + neg[m.end():])
        out.add(neg[:m.start()] + f'there {m.group(1)} an' + neg[m.end():])
    return out


def test_bank_negations_are_clean():
    assert len(BANK) >= 300
    for c in BANK:
        for x, nx in (('P', 'notP'), ('Q', 'notQ')):
            assert c[x] in positive(c[nx]), (c['id'], c[x], c[nx])
            assert not re.search(r"\b(not|no|never)\b|n't", c[x]), (c['id'], c[x])
            assert c[nx].count(' not ') + c[nx].count(' no ') == 1, (c['id'], c[nx])
        assert not re.search(r"\b(I|you|we|my|your|our)\b", c['P'] + ' ' + c['Q']), c['id']


def test_well_formed_and_mix():
    sizes, doms, pos = collections.Counter(), collections.Counter(), collections.Counter()
    for s in got():
        assert sum(s.correct) == 1
        assert len(s.options) in (3, 4) and len(set(s.options)) == len(s.options), s.options
        blob = s.question + ' | ' + ' | '.join(s.options)
        assert not PLACEHOLDER.search(blob), blob
        assert s.metadata['type'] == 'other' and s.metadata['difficulty_level'] == 2
        sizes[len(s.options)] += 1
        doms[s.metadata['domain']] += 1
        if len(s.options) == 3:
            pos[s.answer_index] += 1
    assert sizes[3] > 0.8 * N and sizes[4] > 0.02 * N, sizes
    assert set(doms) == {'daily_life', 'finance', 'medicine'}, doms
    total = sum(pos.values())
    assert all(abs(pos[i] / total - 1 / 3) < 4 * ((2 / 9) / total) ** 0.5 for i in range(3)), pos


def test_reproducible():
    a = sample('equiv_contrapositive', 40, seed=5)
    b = sample('equiv_contrapositive', 40, seed=5)
    assert [(s.question, s.options) for s in a] == [(s.question, s.options) for s in b]


def test_audit_solver_mostly_agrees():
    spec = importlib.util.spec_from_file_location('audit_logic', os.path.join(ROOT, 'tests/solvers/steer/solvers/logic_tom.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = pd.DataFrame({'question_text': [s.question for s in got()], 'options': [list(s.options) for s in got()]})
    res = mod.solve_contrapositive(df)
    ok = sum(1 for p, st, s in zip(res.pred, res.status, got()) if st == 'ok' and p == s.answer_index)
    wrong = sum(1 for p, st, s in zip(res.pred, res.status, got()) if st == 'ok' and p != s.answer_index)
    # the audit solver is a word-overlap heuristic: it may fail to parse, but it must not contradict the key
    assert wrong == 0 and ok > 0.8 * len(got()), (ok, wrong, collections.Counter(res.status))

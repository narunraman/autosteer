"""copeland_method: the key is the Copeland winner, recomputed from the ballots or the one-on-one table in the text.

Independent of the generator: the ballots ("12 members voted for B > C > D") or the one-on-one results
("Book B vs Book C: 41 to 37") are parsed from the question; every pair of alternatives is compared, a win scores 1
and a tie 1/2, and the alternative with the highest total must be unique and be the marked option. For profiles the
one-on-one counts are recomputed from the ballots. Checks also that both types and both tags occur, that three, four
and five alternatives occur, that the Copeland winner is often not a Condorcet winner (and, for profiles, often not
the plurality or Borda winner), and that sampling is reproducible from the seed.
"""
import collections
import re
from itertools import combinations

from autosteer.filler_tools import code_templates as ct
from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'copeland_method'
BALLOT = re.compile(r'^(\d+) members? voted for ([A-Z](?: > [A-Z])+)$', re.M)
ROW = re.compile(r'^\w+ ([A-Z]) vs \w+ ([A-Z]): (\d+) to (\d+)$', re.M)


def contests_from_ballots(ballots):
    alts = ballots[0][1]
    wins = {}
    for x, y in combinations(alts, 2):
        nx = sum(n for n, r in ballots if r.index(x) < r.index(y))
        ny = sum(n for n, r in ballots if r.index(y) < r.index(x))
        wins[(x, y)] = (nx, ny)
    return sorted(alts), wins


def copeland(alts, wins):
    score = collections.defaultdict(float)
    for (x, y), (nx, ny) in wins.items():
        if nx > ny:
            score[x] += 1
        elif ny > nx:
            score[y] += 1
        else:
            score[x] += 0.5
            score[y] += 0.5
    return {a: score[a] for a in alts}


def test_templates():
    assert len(ct.templates(E)) == 6 * 3 * 2 * 2


def test_keys_are_copeland_winners():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(800):
        (row,) = build_rows(s.sample_question(E, 61000 + seed), 'test')
        text = row['question_text']
        if row['type'] == 'profile':
            ballots = [(int(n), r.split(' > ')) for n, r in BALLOT.findall(text)]
            assert ballots and not ROW.search(text)
            alts, wins = contests_from_ballots(ballots)
            first = collections.Counter()
            borda = collections.Counter()
            for n, r in ballots:
                first[r[0]] += n
                for i, c in enumerate(r):
                    borda[c] += n * (len(r) - 1 - i)
        else:
            assert not BALLOT.search(text)
            wins = {(x, y): (int(a), int(b)) for x, y, a, b in ROW.findall(text)}
            alts = sorted({c for pair in wins for c in pair})
            assert len(wins) == len(alts) * (len(alts) - 1) // 2
        sc = copeland(alts, wins)
        top = max(sc.values())
        winners = [a for a in alts if sc[a] == top]
        assert len(winners) == 1, (text, sc)
        w = winners[0]
        labels = [o.split()[-1] for o in row['options']]
        assert sorted(labels) == alts
        assert labels[row['correct_index']] == w, (text, row['options'], sc)
        condorcet = all((wins[(w, y)][0] > wins[(w, y)][1]) if (w, y) in wins else (wins[(y, w)][1] > wins[(y, w)][0])
                        for y in alts if y != w)
        seen['condorcet' if condorcet else 'not_condorcet'] += 1
        if row['type'] == 'profile':
            if w != max(first, key=first.get) and w != max(borda, key=borda.get):
                seen['not_plurality_or_borda'] += 1
        seen[len(alts)] += 1
        seen[row['type']] += 1
    assert seen['profile'] and seen['table'] and seen[3] and seen[4] and seen[5]
    assert 0.45 <= seen['not_condorcet'] / 800 <= 0.75, seen
    assert seen['not_plurality_or_borda'] >= 0.5 * seen['profile'], seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q

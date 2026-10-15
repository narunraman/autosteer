"""auctions_affiliated: keys checked against a revenue table by environment and format.

Independent of the generator: the environment is read from the story's subject (oil tract, returned laptops, wine
lot, forest tract: a common resale value with private noisy estimates, risk-neutral; private dinner, walk-on role:
independent private values, risk-neutral; signed ball, ferry cabin: independent private values with bidders who
prefer a sure $50 to an even chance of $100). Each "Format k: ..." description is recognised by its wording. The
table (higher = more expected revenue):
    affiliated   rising clock 3, second-price 2, first-price 1, falling clock 1   (linkage principle)
    ipv_neutral  all equal                                                       (revenue equivalence)
    ipv_averse   first-price and falling clock 2, second-price and rising clock 1
The keyed ordering must be the table's ordering of the listed formats (ties as "="), no other option may be; the
publish-the-appraisal key is Yes under affiliation and "no difference" otherwise. Also: the stories never name the
environment, every type and environment occurs, and seeded reproducibility.
"""
import collections
import re

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'auctions_affiliated'
ENV = {'oil tract': 'affiliated', 'returned laptops': 'affiliated', 'vintage wine': 'affiliated',
       'forest tract': 'affiliated', 'private dinner': 'ipv_neutral', 'walk-on role': 'ipv_neutral',
       'signed match ball': 'ipv_averse', 'private cabin': 'ipv_averse'}
TABLE = {'affiliated': {'E': 3, 'S': 2, 'F': 1, 'D': 1}, 'ipv_neutral': {'E': 1, 'S': 1, 'F': 1, 'D': 1},
         'ipv_averse': {'E': 1, 'S': 1, 'F': 2, 'D': 2}}
from autosteer.filler_tools.auctions_affiliated_tools import APPRAISAL, ENVIRONMENT  # noqa: E402

BANNED = re.compile(r'affiliat|common value|private value|risk[- ](averse|neutral)|revenue equivalence|linkage|'
                    r'English|Dutch|Vickrey|first-price|second-price', re.I)


def fmt_of(desc):
    if 'starts low and rises' in desc:
        return 'E'
    if 'starts high and falls' in desc:
        return 'D'
    if 'pays the second-highest bid' in desc:
        return 'S'
    assert 'pays their own bid' in desc, desc
    return 'F'


def parse_order(o):
    return [set(int(x) for x in re.findall(r'Format (\d)', tier)) for tier in o.split(' > ')]


def test_keys_by_table():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(500):
        (row,) = build_rows(s.sample_question(E, 76000 + seed), 'test')
        text = row['question_text']
        envs = [e for k, e in ENV.items() if k in text]
        assert len(envs) == 1, text
        env = envs[0]
        story = text.split('Format 1:')[0].split('The seller will use')[0]
        # the stated information structure (ENVIRONMENT / APPRAISAL) uses the standard term "strictly affiliated" as
        # the premise (re-review, October 2026); the rest of the story names nothing
        for s_ in list(ENVIRONMENT.values()) + list(APPRAISAL.values()):
            story = story.replace(s_, '')
        assert not BANNED.search(story), story
        key = row['options'][row['correct_index']]
        seen[(row['type'], env)] += 1
        if row['type'] == 'info_policy':
            want = 'Yes' if env == 'affiliated' else 'It makes no difference'
            assert key.startswith(want), (env, key)
            continue
        formats = {int(k): fmt_of(d) for k, d in re.findall(r'^Format (\d): (.+)$', text, re.M)}
        assert len(formats) == int(row['type'][-1]) == len(set(formats.values()))
        score = {k: TABLE[env][f] for k, f in formats.items()}
        want = [{k for k in score if score[k] == v} for v in sorted(set(score.values()), reverse=True)]
        matches = [o for o in row['options'] if parse_order(o) == want]
        assert matches == [key], (env, formats, row['options'], key)
    for t in ('ranking3', 'ranking4', 'info_policy'):
        for env in TABLE:
            assert seen[(t, env)], seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q

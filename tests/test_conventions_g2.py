"""Review (October 2026), behavioural elements: conventions stored in their templates (formerly load-time fixes).

Code-template fixes have their tests in the element's own test file (sunk_cost, loss_aversion, endowment_effect,
ambiguity_aversion, independence)."""
import re

import pytest

from autosteer.sampler import ElementSampler
from conventions import GAMBLERS_FALLACY_DROPS, swap_lottery_order


@pytest.fixture(scope='module')
def sampler():
    return ElementSampler()


def templates(sampler, element):
    d = sampler._load_template(element)
    return d.questions, d.options


def test_gamblers_fallacy_only_asks_about_a_change(sampler):
    q, _ = templates(sampler, 'gamblers_fallacy')
    texts = dict(zip(q['question_id'], q['question_text']))
    assert not set(texts) & set(GAMBLERS_FALLACY_DROPS)
    assert len(texts) >= 95
    change = re.compile(r'increas|decreas|reduc|higher|lower|less|lessen|improve|better|due|overdue|compensate|balance|'
                        r'make|affect|than before|than the preceding|compared|again|now|still|imminent|eight|one next|'
                        r'more (?:likely|probable)', re.I)
    for qid, t in texts.items():
        # "Yes" is right if the question compares a different outcome with the streak outcome
        assert not re.search(r'(different|other|anything but|something other)\b[^?]*\bmore (likely|probable)'
                             r'(?![^?]*than before)', t) or re.search(r'\b(make|increase|higher)', t), (qid, t)
        assert not re.search(r'baccarat|blackjack|bust|\bcards?\b|\bdeal|\bdeck\b|\bhands?\b|poker(?! dice)', t, re.I), \
            (qid, t)
        assert not re.search(r'lottery numbers have|won by an even number', t), (qid, t)  # parity, no number range
        assert not re.search(r'\bthe die\b[^?]*\bseven\b', t), (qid, t)  # a single die cannot show seven
        assert not re.search(r'(?:hit|hits|hitting) (?:the )?jackpot \{streak\}.*(?:more likely|higher|increase|again\b(?!.*less))',
                             t), (qid, t)
        assert change.search(t), (qid, t)


def test_independence_risk_key_is_described_first_or_second(sampler):
    q, _ = templates(sampler, 'independence_risk')
    ids = sorted(q['question_id'].unique())
    assert len(ids) == 2 * 481  # 481 distinct templates (the data repeated them to balance domains), each in both orders
    originals = q[q['question_id'].isin(ids[:481])]
    for t in originals['question_text']:
        s = swap_lottery_order(t)
        assert s is not None, t
        p = [m.start() for m in re.finditer(r'\{p\}', s)]
        prefix = t[:t.index('{p}')]
        seg_orig = t[t.index('{p}'):t.index('{q}')]
        # the preferred item's lottery (first segment of the original) is now the second one, labelled {first}
        assert s[p[1]:].startswith(seg_orig), (t, s)
        assert s.index('{second}') < s.index('{first}') and t.index('{first}') < t.index('{second}'), (t, s)
        assert s[:s.index('{p}')].replace('{second}', '{first}') == prefix.replace('{second}', '{first}')
    # sampled: the keyed label's lottery comes first about half the time
    first = 0
    for seed in range(300):
        qq = sampler.sample_question('independence_risk', 40000 + seed)
        part = qq['parts'][0]
        key = part['options'][part['correct'].index(True)]
        other = next(o for o in part['options'] if o != key)
        first += part['question'].index(key) < part['question'].index(other)
    assert 100 <= first <= 200, first


@pytest.mark.parametrize('element', ['certainty_effect', 'reflection_effect'])
def test_stated_prior_states_unchanged_preferences(sampler, element):
    q, _ = templates(sampler, element)
    sp = q[q['type'].astype(str).str.strip() == 'stated_prior']
    assert len(sp) == 460 if element == 'certainty_effect' else len(sp) == 260
    need = 'attitude to risk has not changed since then' if element == 'reflection_effect' else 'not changed since then'
    for t in sp['question_text']:
        assert need in t and t.rstrip().endswith('?'), t


def test_reflection_loss_options_are_payments(sampler):
    q, o = templates(sampler, 'reflection_effect')
    types = dict(zip(q['question_id'], q['type'].astype(str).str.strip()))
    for qid, sid, text in zip(o['question_id'], o['sub_id'], o['option_text']):
        loss = types[qid] == 'stated_prior' or int(sid) == 1
        if loss:
            assert 'chance of paying' not in text, text
    for seed in range(40):
        qq = sampler.sample_question('reflection_effect', 41000 + seed)
        loss_part = qq['parts'][-1]
        assert not any('chance of paying' in x for x in loss_part['options']), loss_part



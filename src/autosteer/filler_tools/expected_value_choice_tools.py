"""maximize_expected_utility (STEER 2.4.b): choose between a sure amount and lotteries with utility linear in money.

Decision of 3 October 2026: keep only the plain ("regular") type. Each question states that the chooser's utility
is linear in money, so the expected-utility maximiser picks the option with the highest expected value. The
certainty, reflection and loss-aversion types are not generated here (certainty and reflection are their own
elements; loss aversion is held).

Templates are written here. The scenario sentences are the released ICML stems of the regular type (game shows,
casinos, contests; job offers with a chance of a bonus), recovered from the released rows; the options and the
sentence about utility are new. Options:
  gambling: "A guaranteed $76" and one or two lotteries "$118 with a 35% chance and $52 with a 65% chance";
  job_offer: "Job P" (a sure salary) and "Job Q" (a lower salary with a chance of a bonus).
Rules: probabilities are whole percentages (10-90); every lottery has one outcome above and one below the sure
amount, so no option dominates another outright; the best expected value beats every other option's by at least 5%
(gambling) or 2% (job offers). The key is the option with the highest expected value.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

UTILITY = 'Your utility is linear in money: every dollar is worth the same to you, however much you already have.'

# Released stems of the regular type (gambling domain) that name two options.
TWO_STEMS = [
    'You are at a game show and have to choose between two final prize options:',
    'You are at a game show and have to choose between two final prizes:',
    'You are at a game show and must choose between two final prize options:',
    'You are at a game show and must choose between two doors:',
    'You are at a game show and have to make a choice between two options for your prize:',
    'You are at a casino and have to choose between two slot machines for a single play:',
    'You are participating in a game show and have to choose between two final prize options:',
    'You are at a casino facing two betting options on a game:',
    'You are at a casino with the option to play one of two games:',
    'You are at a game show and must pick between two doors:',
    'You enter a contest and make it to the final round where you must choose one of these two options:',
    'You are at a casino and must pick between two slot machines to play:',
    "You're at a game show and must choose between two prize doors:",
    'You are offered a chance to participate in a game show. To win a prize, you must choose one of two options:',
    'You are at a game show and must select one of two curtains. Behind each curtain, there is a prize with different values:',
    'A contestant on a game show must choose between two final prize options:',
]
# Released stems that do not name the number of options (used with two and with three options).
ANY_STEMS = [
    'You need to choose one of the following:',
    'You are faced with a gambling decision in a game show:',
    "You've been given a chance to participate in a game show where you need to choose one of the following:",
    'You are participating in a game show and need to choose one of the following prizes:',
    "Imagine you're playing a game at a fair. You have one free ticket to play, and you need to choose one of the following:",
    'You have entered a game show and reached the final stage. To win a prize, you need to choose one of the following:',
]
# Released job-offer stem (regular type), with three letter pairs.
JOB_STEM = ('You currently have two job offers. Job {a} offers a salary of {{salary_sure}} per year with no chance of a '
            'bonus. Job {b} offers a salary of {{salary_risky}} per year but with a {{bonus_chance}} chance of a '
            '{{bonus}} bonus.')
JOB_LETTERS = [('P', 'Q'), ('E', 'F'), ('G', 'H')]


def _sentence(stem: str) -> str:
    return stem[:-1] + '.' if stem.endswith(':') else stem


def _templates() -> list[ct.CodeTemplate]:
    out = []
    ask = ' Which option maximizes your expected utility?'
    for stem in TWO_STEMS:
        out.append(ct.CodeTemplate(parts=(f'{_sentence(stem)} {UTILITY}{ask}',), type='regular', domain='gambling',
                                   tags='second_person;money', difficulty_level=3,
                                   spec={'kind': 'gamble', 'lotteries': 1}))
    for stem in ANY_STEMS:
        for k in (1, 2):
            out.append(ct.CodeTemplate(parts=(f'{_sentence(stem)} {UTILITY}{ask}',) if k == 1 else
                                       (f'{_sentence(stem)} {UTILITY} Which of the three options maximizes your '
                                        f'expected utility?',),
                                       type='regular', domain='gambling', tags='second_person;money',
                                       difficulty_level=3 if k == 1 else 4, spec={'kind': 'gamble', 'lotteries': k}))
    for a, b in JOB_LETTERS:
        out.append(ct.CodeTemplate(
            parts=(f'{JOB_STEM.format(a=a, b=b)} {UTILITY} Which job should you choose to maximize your expected '
                   f'utility?',),
            type='regular', domain='job_offer', tags='second_person;money', difficulty_level=4,
            spec={'kind': 'job', 'sure': a, 'risky': b}))
    return out


def _rng():
    return filler_utils.rng


def dollars(v: int) -> str:
    return f'${int(v):,}'


def lottery_text(hi: int, p: int, lo: int) -> str:
    return f'{dollars(hi)} with a {p}% chance and {dollars(lo)} with a {100 - p}% chance'


def _gamble(n_lotteries: int):
    r = _rng()
    while True:
        sure = int(r.integers(20, 501))
        lots = []
        for _ in range(n_lotteries):
            hi = int(r.integers(sure + 1, 2 * sure + 30))
            lo = int(r.integers(0, sure))
            p = int(r.integers(10, 91))
            lots.append((hi, p, lo))
        evs = [float(sure)] + [(hi * p + lo * (100 - p)) / 100 for hi, p, lo in lots]
        texts = [f'A guaranteed {dollars(sure)}'] + [lottery_text(*x) for x in lots]
        if len(set(texts)) != len(texts):
            continue
        best = max(range(len(evs)), key=lambda i: evs[i])
        if all(evs[best] >= 1.05 * v for i, v in enumerate(evs) if i != best):
            return texts, best


@function_registry.register('maximize_expected_utility')
def fill_maximize_expected_utility(metadata: dict) -> dict:
    spec = ct.spec_for('maximize_expected_utility', metadata['question_text'])
    r = _rng()
    if spec['kind'] == 'gamble':
        options, best = _gamble(spec['lotteries'])
        return {'options_list': [options], 'correct_answer': [[int(i == best) for i in range(len(options))]]}
    while True:
        sure = int(r.integers(50, 121)) * 1000
        risky = int(r.integers(30, sure // 1000)) * 1000
        bonus = int(r.integers((sure - risky) // 1000 + 1, (sure - risky) // 1000 * 3 + 10)) * 1000
        p = int(r.integers(10, 91))
        ev_risky = risky + bonus * p / 100
        if max(sure, ev_risky) >= 1.02 * min(sure, ev_risky):
            break
    options = [f"Job {spec['sure']}", f"Job {spec['risky']}"]
    best = 0 if sure > ev_risky else 1
    return {'salary_sure': dollars(sure), 'salary_risky': dollars(risky), 'bonus_chance': f'{p}%',
            'bonus': dollars(bonus), 'options_list': [options], 'correct_answer': [[int(i == best) for i in (0, 1)]]}


ct.register('maximize_expected_utility', _templates())

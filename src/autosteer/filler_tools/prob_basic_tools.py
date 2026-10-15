"""prob_basic (STEER 1.3.a, Compute Probabilities of Outcomes): simple events, keys computed exactly with fractions.

Generator of October 2026 (the released questions were written one by one by GPT-4, with no slots and no computed
answers). Templates are written here; every probability is a Fraction computed from counts (favourable outcomes /
equally likely outcomes). Types:

    same_probability  Two sources state a probability in two different formats (0.4, 40%, 2/5, "2 in 5"); "Do they
                      give the same probability?" Yes in half of the questions; otherwise the second value differs by
                      0.05 to 0.15, or is a tenth of the first (0.4 vs 4%). Contexts: weather, sports, games, health.
    most_likely       Which of two or three events is most likely. Domain mixed: events with a fair die, two fair
                      dice, fair coins, a standard 52-card deck or a bag of marbles, each described on its own;
                      domain urns: two or three bags with stated marble counts, "from which bag is a red marble
                      most likely?". The probabilities are distinct, the best at least 0.02 ahead of the next.
    least_likely      The same, least likely.
    compute           The probability of one event (one die, two dice, two or three coin flips, one card, one marble
                      from a bag with stated counts), four options as reduced fractions: the key and three distinct
                      wrong values from common mistakes (the complement, favourable over unfavourable, one count off,
                      double or half) filled up with nearby fractions.
Decimals have at most two places (the sampler rounds decimals to two places).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

ELEMENT = 'prob_basic'

# Fixed events: family -> (setup sentences for compute, [(standalone description, clause after "that", k, n)]).
# Each family has two setup sentences and at least 10 events, so every compute cell (type, domain, difficulty) has at
# least 20 distinct question texts (the export target; added October 2026, the first lists gave 6 to 14).
EVENTS = {
    'die': (('A fair six-sided die is rolled once.', 'Someone rolls a fair six-sided die once.'), [
        ('rolling a 6 with one fair six-sided die', 'the die shows a 6', 1, 6),
        ('rolling an even number with one fair six-sided die', 'the die shows an even number', 3, 6),
        ('rolling a number greater than 4 with one fair six-sided die', 'the die shows a number greater than 4', 2, 6),
        ('rolling a number less than 5 with one fair six-sided die', 'the die shows a number less than 5', 4, 6),
        ('rolling a multiple of 3 with one fair six-sided die', 'the die shows a multiple of 3', 2, 6),
        ('rolling anything but a 6 with one fair six-sided die', 'the die does not show a 6', 5, 6),
        ('rolling a 1 with one fair six-sided die', 'the die shows a 1', 1, 6),
        ('rolling an odd number with one fair six-sided die', 'the die shows an odd number', 3, 6),
        ('rolling a number greater than 2 with one fair six-sided die', 'the die shows a number greater than 2', 4, 6),
        ('rolling a number less than 3 with one fair six-sided die', 'the die shows a number less than 3', 2, 6),
        ('rolling a 5 or a 6 with one fair six-sided die', 'the die shows a 5 or a 6', 2, 6),
        ('rolling a prime number with one fair six-sided die', 'the die shows a prime number', 3, 6),
        ('rolling anything but a 1 with one fair six-sided die', 'the die does not show a 1', 5, 6),
    ]),
    'two_dice': (('Two fair six-sided dice are rolled.', 'A pair of fair six-sided dice is rolled once.'), [
        ('rolling a total of 7 with two fair six-sided dice', 'the total is 7', 6, 36),
        ('rolling a total of 12 with two fair six-sided dice', 'the total is 12', 1, 36),
        ('rolling a total greater than 9 with two fair six-sided dice', 'the total is greater than 9', 6, 36),
        ('rolling a total of 2 or 3 with two fair six-sided dice', 'the total is 2 or 3', 3, 36),
        ('rolling two even numbers with two fair six-sided dice', 'both dice show an even number', 9, 36),
        ('rolling at least one 6 with two fair six-sided dice', 'at least one die shows a 6', 11, 36),
        ('rolling a total of 8 with two fair six-sided dice', 'the total is 8', 5, 36),
        ('rolling the same number on both of two fair six-sided dice', 'both dice show the same number', 6, 36),
        ('rolling a total less than 5 with two fair six-sided dice', 'the total is less than 5', 6, 36),
        ('rolling a total of 11 with two fair six-sided dice', 'the total is 11', 2, 36),
        ('rolling no 6 on either of two fair six-sided dice', 'neither die shows a 6', 25, 36),
        ('rolling an odd total with two fair six-sided dice', 'the total is odd', 18, 36),
    ]),
    'two_coins': (('A fair coin is flipped twice.', 'A fair coin is flipped two times in a row.'), [
        ('getting two heads in two flips of a fair coin', 'both flips are heads', 1, 4),
        ('getting at least one head in two flips of a fair coin', 'at least one flip is heads', 3, 4),
        ('getting exactly one head in two flips of a fair coin', 'exactly one flip is heads', 2, 4),
        ('getting two tails in two flips of a fair coin', 'both flips are tails', 1, 4),
        ('getting at least one tail in two flips of a fair coin', 'at least one flip is tails', 3, 4),
        ('getting heads first and then tails in two flips of a fair coin',
         'the first flip is heads and the second is tails', 1, 4),
        ('getting the same side twice in two flips of a fair coin', 'both flips show the same side', 2, 4),
        ('getting heads on the second of two flips of a fair coin', 'the second flip is heads', 2, 4),
        ('getting at most one head in two flips of a fair coin', 'at most one flip is heads', 3, 4),
        ('getting exactly one tail in two flips of a fair coin', 'exactly one flip is tails', 2, 4),
        ('getting tails on the first of two flips of a fair coin', 'the first flip is tails', 2, 4),
    ]),
    'three_coins': (('A fair coin is flipped three times.', 'A fair coin is flipped three times in a row.'), [
        ('getting three heads in three flips of a fair coin', 'all three flips are heads', 1, 8),
        ('getting at least one tail in three flips of a fair coin', 'at least one flip is tails', 7, 8),
        ('getting exactly two heads in three flips of a fair coin', 'exactly two flips are heads', 3, 8),
        ('getting three tails in three flips of a fair coin', 'all three flips are tails', 1, 8),
        ('getting at least one head in three flips of a fair coin', 'at least one flip is heads', 7, 8),
        ('getting exactly one head in three flips of a fair coin', 'exactly one flip is heads', 3, 8),
        ('getting exactly two tails in three flips of a fair coin', 'exactly two flips are tails', 3, 8),
        ('getting at least two heads in three flips of a fair coin', 'at least two flips are heads', 4, 8),
        ('getting heads on the first of three flips of a fair coin', 'the first flip is heads', 4, 8),
        ('getting the same side on all three flips of a fair coin', 'all three flips show the same side', 2, 8),
        ('getting heads on each of the first two of three flips of a fair coin', 'the first two flips are heads', 2, 8),
        ('getting at most one head in three flips of a fair coin', 'at most one flip is heads', 4, 8),
    ]),
    'cards': (('One card is drawn at random from a well-shuffled standard 52-card deck.',
               'A card is dealt at random from a well-shuffled standard deck of 52 playing cards.'), [
        ('drawing a heart from a standard 52-card deck', 'the card is a heart', 13, 52),
        ('drawing an ace from a standard 52-card deck', 'the card is an ace', 4, 52),
        ('drawing a red card from a standard 52-card deck', 'the card is red', 26, 52),
        ('drawing a jack, queen or king from a standard 52-card deck', 'the card is a jack, queen or king', 12, 52),
        ('drawing the queen of spades from a standard 52-card deck', 'the card is the queen of spades', 1, 52),
        ('drawing a black king from a standard 52-card deck', 'the card is a black king', 2, 52),
        ('drawing a card that is not a club from a standard 52-card deck', 'the card is not a club', 39, 52),
        ('drawing a spade from a standard 52-card deck', 'the card is a spade', 13, 52),
        ('drawing a diamond from a standard 52-card deck', 'the card is a diamond', 13, 52),
        ('drawing a king from a standard 52-card deck', 'the card is a king', 4, 52),
        ('drawing a card numbered 2 to 10 from a standard 52-card deck', 'the card is numbered 2 to 10', 36, 52),
        ('drawing a red ace from a standard 52-card deck', 'the card is a red ace', 2, 52),
        ('drawing a black card from a standard 52-card deck', 'the card is black', 26, 52),
        ('drawing a heart or a spade from a standard 52-card deck', 'the card is a heart or a spade', 26, 52),
        ('drawing a card that is not an ace from a standard 52-card deck', 'the card is not an ace', 48, 52),
        ('drawing the ace of hearts from a standard 52-card deck', 'the card is the ace of hearts', 1, 52),
        ('drawing a red jack, queen or king from a standard 52-card deck', 'the card is a red jack, queen or king',
         6, 52),
        ('drawing a 7 from a standard 52-card deck', 'the card is a 7', 4, 52),
    ]),
}
COLOURS = ['red', 'blue', 'green', 'yellow', 'white']

SAME_CONTEXTS = {
    'weather': ('One weather service puts the chance of rain tomorrow at {value_a}. Another service puts the chance '
                'of rain tomorrow at {value_b}.', 'Do the two services give the same probability of rain?'),
    'sports': ("One analyst puts the home team's chance of winning Saturday's match at {value_a}. A second analyst "
               'puts it at {value_b}.', 'Do the two analysts give the home team the same probability of winning?'),
    'games': ('At a fair, the chance of winning a prize at the ring toss is posted as {value_a}. At the duck pond, '
              'the chance of winning a prize is posted as {value_b}.',
              'Are the two chances of winning a prize the same?'),
    'health': ('One clinical study puts the chance of a mild headache after the vaccine at {value_a}. A second study '
               'puts the same chance at {value_b}.', 'Do the two studies give the same probability?'),
}
SAME_STEMS_ALT = {
    'weather': 'Is the probability of rain the same in both forecasts?',
    'sports': "Is the home team's probability of winning the same in both estimates?",
    'games': 'Is the probability of winning a prize equal at the two stalls?',
    'health': 'Is the stated probability the same in both studies?',
}


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for domain, (body, q) in SAME_CONTEXTS.items():
        for k, stem in enumerate((q, SAME_STEMS_ALT[domain])):
            out.append(ct.CodeTemplate(parts=(f'{body} {stem}',), type='same_probability', domain=domain,
                                       tags='', difficulty_level=0, spec={'kind': 'same', 'domain': domain}))
    for direction, word, chance in (('most_likely', 'most', 'highest'), ('least_likely', 'least', 'lowest')):
        for stem in (f'Each event below is a single, separate chance experiment. Which event is {word} likely?',
                     f'Which of the following events is {word} likely to happen?'):
            out.append(ct.CodeTemplate(parts=(stem,), type=direction, domain='mixed', tags='', difficulty_level=1,
                                       spec={'kind': 'compare_events', 'direction': direction}))
        for stem in (f'One marble is drawn at random from a bag. {{bags}} From which bag is drawing a red marble '
                     f'{word} likely?',
                     f'{{bags}} A marble is drawn at random from one of the bags. Which bag gives the {chance} '
                     f'chance of drawing a red marble?'):
            out.append(ct.CodeTemplate(parts=(stem,), type=direction, domain='urns', tags='', difficulty_level=1,
                                       spec={'kind': 'compare_bags', 'direction': direction}))
    for family, (setups, _) in EVENTS.items():
        for setup in setups:
            out.append(ct.CodeTemplate(parts=(f'{setup} What is the probability that {{event}}?',), type='compute',
                                       domain='coins' if 'coins' in family else 'dice' if 'di' in family else family,
                                       tags='', difficulty_level=1 if family in ('two_dice', 'three_coins') else 0,
                                       spec={'kind': 'compute', 'family': family}))
    out.append(ct.CodeTemplate(parts=('{bag} One marble is drawn at random. What is the probability that it is '
                                      '{colour}?',), type='compute', domain='urns', tags='', difficulty_level=0,
                               spec={'kind': 'compute', 'family': 'urn'}))
    return out


def frac(p: Fraction) -> str:
    return f'{p.numerator}/{p.denominator}'


def render(p: Fraction, fmt: str) -> str:
    """p has denominator dividing 100 for 'decimal' and 'percent'."""
    if fmt == 'decimal':
        s = f'{float(p):.2f}'.rstrip('0')
        return s
    if fmt == 'percent':
        c = p * 100
        assert c.denominator == 1
        return f'{c.numerator}%'
    if fmt == 'fraction':
        return frac(p)
    return f'{p.numerator} in {p.denominator}'


def representable(p: Fraction) -> bool:
    return 0 < p < 1 and (p * 100).denominator == 1


def bag_text(counts: dict[str, int]) -> str:
    parts = [f'{n} {c}' for c, n in counts.items()]
    return (', '.join(parts[:-1]) + ' and ' + parts[-1]) + ' marbles'


def draw_bag(r, colours: int = None) -> dict[str, int]:
    k = colours or int(r.integers(2, 4))
    names = ['red'] + [COLOURS[1:][i] for i in sorted(r.choice(len(COLOURS) - 1, size=k - 1, replace=False))]
    return {c: int(r.integers(2, 10)) for c in names}  # at least 2: '1 yellow marbles' would read badly


def wrong_values(r, k: int, n: int) -> list[Fraction]:
    p = Fraction(k, n)
    cands = [1 - p]
    if n - k > 0:
        cands.append(Fraction(k, n - k))
    cands += [Fraction(k + 1, n), Fraction(k - 1, n) if k > 1 else Fraction(k, n + 1), p / 2, p * 2,
              Fraction(1, n), Fraction(k, n + 1)]
    out = []
    order = r.permutation(len(cands))
    for i in order:
        c = cands[int(i)]
        if 0 < c < 1 and c != p and c not in out:
            out.append(c)
        if len(out) == 3:
            return out
    d = 2
    while len(out) < 3:  # nearby fractions
        for num in range(1, d):
            c = Fraction(num, d)
            if c != p and c not in out and len(out) < 3:
                out.append(c)
        d += 1
    return out


@function_registry.register(ELEMENT)
def fill_prob_basic(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    r = filler_utils.rng
    kind = spec['kind']
    out = {}
    if kind == 'same':
        while True:
            p = Fraction(int(r.integers(1, 20)) * 5, 100)
            fa, fb = (['decimal', 'percent', 'fraction', 'in'][int(i)] for i in r.choice(4, size=2, replace=False))
            same = bool(r.integers(0, 2))
            if same:
                q = p
            elif r.random() < 0.25 and (p * 100).numerator % 10 == 0:
                q = p / 10
            else:
                q = p + Fraction(int(r.choice([-15, -10, -5, 5, 10, 15])), 100)
            if representable(q):
                break
        out.update(value_a=render(p, fa), value_b=render(q, fb))
        options = ['Yes', 'No']
        key = 'Yes' if p == q else 'No'
    elif kind == 'compare_events':
        while True:
            m = int(r.integers(2, 4))
            pool = [(fam, e) for fam, (_, evs) in EVENTS.items() for e in evs]
            pick = [pool[int(i)] for i in r.choice(len(pool), size=m, replace=False)]
            if len({fam for fam, _ in pick}) < m:
                continue  # one event per family: each option is a different experiment
            probs = [Fraction(e[2], e[3]) for _, e in pick]
            srt = sorted(probs, reverse=spec['direction'] == 'most_likely')
            if len(set(probs)) == m and abs(srt[0] - srt[1]) >= Fraction(2, 100):
                break
        options = [e[0][0].upper() + e[0][1:] for _, e in pick]
        key = options[probs.index(srt[0])]
    elif kind == 'compare_bags':
        while True:
            m = int(r.integers(2, 4))
            bags = [draw_bag(r, 2) for _ in range(m)]
            probs = [Fraction(b['red'], sum(b.values())) for b in bags]
            srt = sorted(probs, reverse=spec['direction'] == 'most_likely')
            if len(set(probs)) == m and abs(srt[0] - srt[1]) >= Fraction(2, 100):
                break
        labels = 'ABC'[:m]
        out['bags'] = ' '.join(f'Bag {x} holds {bag_text(b)}.' for x, b in zip(labels, bags))
        options = [f'Bag {x}' for x in labels]
        key = options[probs.index(srt[0])]
    else:
        if spec['family'] == 'urn':
            bag = draw_bag(r)
            colour = list(bag)[int(r.integers(0, len(bag)))]
            k, n = bag[colour], sum(bag.values())
            out.update(bag=f'A bag holds {bag_text(bag)}.', colour=colour)
        else:
            evs = EVENTS[spec['family']][1]
            _, clause, k, n = evs[int(r.integers(0, len(evs)))]
            out['event'] = clause
        p = Fraction(k, n)
        # Fixed events: the distractors depend only on the event (k, n), so a repeated question text always comes
        # with the same options and the exporter sees it as a duplicate, not as a new question.
        wr = r if spec['family'] == 'urn' else np.random.default_rng([k, n])
        options = [frac(p)] + [frac(x) for x in wrong_values(wr, k, n)]
        key = options[0]
    out['options_list'] = [options]
    out['correct_answer'] = [[int(o == key) for o in options]]
    return out


ct.register(ELEMENT, _templates())

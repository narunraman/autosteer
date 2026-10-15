"""prob_basic: every key recomputed from the question text by enumerating the sample space, independently of the
filler's hand-entered counts.

Event wording is read with regular expressions (numbers parsed from the text) and evaluated by enumerating the
outcomes: 6 faces, 36 ordered pairs of dice, 4 or 8 coin sequences, a 52-card deck built here, or the marbles of
the stated bag. Checks on 1,500 seeded samples:
  same_probability  both stated values parsed (0.35, 35%, 7/20, "7 in 20"), in two different formats; key Yes iff
                    they are equal as fractions;
  most / least      each option's probability computed; the key has the strictly largest (smallest) one;
  compute           the key is the exact probability, written as a reduced fraction; options distinct values;
plus every type and domain occurs, both Yes and No occur, and the same seed gives the same question.
"""
import collections
import itertools
import re
from fractions import Fraction

from helpers import sample
from autosteer.sampler import ElementSampler

SUITS = ['hearts', 'diamonds', 'clubs', 'spades']
DECK = [(rank, suit) for rank in range(1, 14) for suit in SUITS]


def prob(space, pred):
    space = list(space)
    return Fraction(sum(1 for w in space if pred(w)), len(space))


def die_pred(t):
    m = re.search(r'a (\d) or a (\d)', t)
    if m:
        return lambda x: x in (int(m.group(1)), int(m.group(2)))
    m = re.search(r'greater than (\d+)', t)
    if m:
        return lambda x: x > int(m.group(1))
    m = re.search(r'less than (\d+)', t)
    if m:
        return lambda x: x < int(m.group(1))
    if re.search(r'anything but a (\d)|does not show a (\d)', t):
        v = int(re.search(r'(?:anything but a|does not show a) (\d)', t).group(1))
        return lambda x: x != v
    m = re.search(r'multiple of (\d+)', t)
    if m:
        return lambda x: x % int(m.group(1)) == 0
    if 'even number' in t:
        return lambda x: x % 2 == 0
    if 'odd number' in t:
        return lambda x: x % 2 == 1
    if 'prime number' in t:
        return lambda x: x in (2, 3, 5)
    m = re.search(r'(?:rolling a|shows a) (\d)\b', t)
    return lambda x: x == int(m.group(1))


def two_dice_pred(t):
    m = re.search(r'total (?:of|is) (\d+) or (\d+)', t)
    if m:
        return lambda d: sum(d) in (int(m.group(1)), int(m.group(2)))
    m = re.search(r'total (?:is )?greater than (\d+)', t)
    if m:
        return lambda d: sum(d) > int(m.group(1))
    m = re.search(r'total (?:is )?less than (\d+)', t)
    if m:
        return lambda d: sum(d) < int(m.group(1))
    if re.search(r'odd total|total is odd', t):
        return lambda d: sum(d) % 2 == 1
    m = re.search(r'total (?:of|is) (\d+)', t)
    if m:
        return lambda d: sum(d) == int(m.group(1))
    if 'same number' in t:
        return lambda d: d[0] == d[1]
    if re.search(r'two even numbers|both dice show an even', t):
        return lambda d: d[0] % 2 == 0 and d[1] % 2 == 0
    m = re.search(r'\bno (\d)\b|neither die shows a (\d)', t)
    if m:
        v = int(m.group(1) or m.group(2))
        return lambda d: v not in d
    m = re.search(r'at least one (?:die shows a )?(\d)', t)
    return lambda d: int(m.group(1)) in d


WORDS = {'one': 1, 'two': 2, 'three': 3}


def coin_pred(t):
    def side(w):
        return 'H' if w.startswith('head') else 'T'
    if 'same side' in t:
        return lambda s: len(set(s)) == 1
    if re.search(r'heads first and then tails|first flip is heads and the second is tails', t):
        return lambda s: s[:2] == 'HT'
    if re.search(r'first two flips are heads|heads on each of the first two', t):
        return lambda s: s[:2] == 'HH'
    m = re.search(r'(heads|tails) on the (first|second) of', t) or re.search(r'the (first|second) flip is (heads|tails)', t)
    if m:
        g = m.groups()
        sd, pos = (g[0], g[1]) if g[0] in ('heads', 'tails') else (g[1], g[0])
        i = 0 if pos == 'first' else 1
        return lambda s: s[i] == side(sd)
    m = re.search(r'(at least|at most|exactly) (one|two) (?:flips? (?:is|are) )?(head|tail)', t)
    if m:
        k, c = WORDS[m.group(2)], side(m.group(3))
        op = {'at least': lambda a, b: a >= b, 'at most': lambda a, b: a <= b, 'exactly': lambda a, b: a == b}
        f = op[m.group(1)]
        return lambda s: f(s.count(c), k)
    m = re.search(r'(?:two|three) (heads|tails) in|(?:both|all three) flips are (heads|tails)', t)
    c = side(m.group(1) or m.group(2))
    return lambda s: set(s) == {c}


def card_pred(t):
    if 'ace of hearts' in t:
        return lambda c: c == (1, 'hearts')
    if 'queen of spades' in t:
        return lambda c: c == (12, 'spades')
    if 'black king' in t:
        return lambda c: c[0] == 13 and c[1] in ('clubs', 'spades')
    if 'red jack, queen or king' in t:
        return lambda c: c[0] >= 11 and c[1] in ('hearts', 'diamonds')
    if 'jack, queen or king' in t:
        return lambda c: c[0] >= 11
    if 'not a club' in t:
        return lambda c: c[1] != 'clubs'
    if 'not an ace' in t:
        return lambda c: c[0] != 1
    if 'red ace' in t:
        return lambda c: c[0] == 1 and c[1] in ('hearts', 'diamonds')
    m = re.search(r'numbered (\d+) to (\d+)', t)
    if m:
        return lambda c: int(m.group(1)) <= c[0] <= int(m.group(2))
    if 'heart or a spade' in t:
        return lambda c: c[1] in ('hearts', 'spades')
    if re.search(r'\ba heart\b', t):
        return lambda c: c[1] == 'hearts'
    if re.search(r'\ba spade\b', t):
        return lambda c: c[1] == 'spades'
    if re.search(r'\ba diamond\b', t):
        return lambda c: c[1] == 'diamonds'
    if re.search(r'\ban ace\b', t):
        return lambda c: c[0] == 1
    if re.search(r'\ba king\b', t):
        return lambda c: c[0] == 13
    if re.search(r'red card|card is red', t):
        return lambda c: c[1] in ('hearts', 'diamonds')
    if re.search(r'black card|card is black', t):
        return lambda c: c[1] in ('clubs', 'spades')
    m = re.search(r'\ba (\d+)\b', t)
    if m:
        return lambda c: c[0] == int(m.group(1))
    raise AssertionError(t)


def event_prob(t, setup=''):
    t = t.lower()
    ctx = (setup + ' ' + t).lower()
    if 'two fair six-sided dice' in ctx or 'pair of fair six-sided dice' in ctx:
        return prob(itertools.product(range(1, 7), repeat=2), two_dice_pred(t))
    if 'die' in ctx:
        return prob(range(1, 7), die_pred(t))
    if 'coin' in ctx:
        n = 3 if re.search(r'three (?:flips|times)', ctx) else 2
        return prob((''.join(x) for x in itertools.product('HT', repeat=n)), coin_pred(t))
    if 'card' in ctx:
        return prob(DECK, card_pred(t))
    raise AssertionError(t)


BAG = re.compile(r'((?:\d+ \w+(?:, | and ))*\d+ \w+) marbles')


def bag_counts(text):
    return {c: int(n) for n, c in re.findall(r'(\d+) (\w+)', text)}


VALUE = re.compile(r'(?:at|as) (\d+ in \d+|\d+/\d+|\d+%|0\.\d+)(?=[ .])')


def parse_value(v):
    if v.endswith('%'):
        return Fraction(int(v[:-1]), 100), 'percent'
    if ' in ' in v:
        a, b = v.split(' in ')
        return Fraction(int(a), int(b)), 'in'
    if '/' in v:
        a, b = v.split('/')
        return Fraction(int(a), int(b)), 'fraction'
    return Fraction(v), 'decimal'


def test_keys_by_enumeration():
    seen = collections.Counter()
    for s in sample('prob_basic', 1500, seed=4505):
        assert len(set(s.options)) == len(s.options) >= 2
        t = s.type
        if t == 'same_probability':
            vals = [parse_value(v) for v in VALUE.findall(s.question)]
            assert len(vals) == 2 and vals[0][1] != vals[1][1], s.question
            assert s.answer == ('Yes' if vals[0][0] == vals[1][0] else 'No'), s.question
            seen[('same', s.answer)] += 1
        elif t in ('most_likely', 'least_likely'):
            if s.metadata['domain'] == 'urns':
                bags = dict(re.findall(r'Bag ([ABC]) holds ([^.]*)\.', s.question))
                probs = []
                for o in s.options:
                    c = bag_counts(bags[o[-1]])
                    probs.append(Fraction(c['red'], sum(c.values())))
            else:
                probs = [event_prob(o) for o in s.options]
            best = (max if t == 'most_likely' else min)(probs)
            assert probs.count(best) == 1 and s.answer_index == probs.index(best), (s.question, s.options, probs)
        else:
            if s.metadata['domain'] == 'urns':
                c = bag_counts(re.search(r'A bag holds ([^.]*)\.', s.question).group(1))
                colour = re.search(r'probability that it is (\w+)\?', s.question).group(1)
                p = Fraction(c[colour], sum(c.values()))
            else:
                setup, clause = re.match(r'(.*?) What is the probability that (.*)\?$', s.question).groups()
                p = event_prob(clause, setup)
            vals = [Fraction(o) for o in s.options]
            assert all(re.fullmatch(r'\d+/\d+', o) and Fraction(o).denominator == int(o.split('/')[1])
                       for o in s.options), s.options  # reduced fractions
            assert len(set(vals)) == len(vals) == 4 and all(0 < v < 1 for v in vals)
            assert vals.count(p) == 1 and s.answer_index == vals.index(p), (s.question, s.options, p)
        seen[t] += 1
        seen[s.metadata['domain']] += 1
    for k in ['same_probability', 'most_likely', 'least_likely', 'compute', ('same', 'Yes'), ('same', 'No'),
              'weather', 'sports', 'games', 'health', 'mixed', 'urns', 'dice', 'coins', 'cards']:
        assert seen[k] > 15, (k, seen)


def test_seeded_reproducibility():
    s = ElementSampler()
    for seed in range(20):
        assert s.sample_question('prob_basic', 61 + seed) == s.sample_question('prob_basic', 61 + seed)

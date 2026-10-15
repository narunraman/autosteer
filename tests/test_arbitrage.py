"""arbitrage: buy where the good is cheap, ship it, sell where it is dear.

Recomputed from the prices and transport cost shown in the question: profit per unit = price_b - price_a - cost.
If it is positive the key is "Yes, for a profit of $<profit> per unit."; otherwise there must be no arbitrage in
either direction (|price_b - price_a| < cost) and the key is "No". Options are distinct, exactly one option
matches the recomputed answer, and the marked option is that one.
"""
import re

from helpers import sample

NO = "No, there's no arbitrage opportunity."
PROFIT = re.compile(r'^Yes, for a profit of \$(\d+(?:\.\d+)?) per unit\.$')


def shown(question, value):
    """The number as the question shows it (2 decimals, trailing zeros dropped)."""
    text = f'{value:.2f}'.rstrip('0').rstrip('.')
    assert re.search(rf'(?<![\d.]){re.escape(text)}(?![\d])', question), (text, question)
    return float(text)


def test_arbitrage():
    seen = set()
    for s in sample('arbitrage', 300, seed=21):
        p = s.params
        price_a, price_b, cost = (shown(s.question, float(p[k])) for k in ('price_a', 'price_b', 'cost'))
        assert len(set(s.options)) == len(s.options) == 4, s.options
        assert s.options.count(NO) == 1
        profits = [float(PROFIT.match(o).group(1)) for o in s.options if o != NO]
        assert len(profits) == 3 and len(set(profits)) == 3, s.options
        profit = round(price_b - price_a - cost, 2)
        if profit > 0:
            expected = [i for i, o in enumerate(s.options) if o != NO and abs(float(PROFIT.match(o).group(1)) - profit) < 0.006]
            # the other profits are not near-twins of the true one
            assert sum(abs(x - profit) < 1 for x in profits) == 1, (profit, s.options)
            seen.add('yes')
        else:
            assert abs(price_b - price_a) < cost, (price_a, price_b, cost)  # no arbitrage in either direction
            expected = [s.options.index(NO)]
            seen.add('no')
        assert expected == [s.answer_index], (s.question, s.options, profit)
    assert seen == {'yes', 'no'}

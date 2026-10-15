# import random
# import numpy as np
from .fill_registry import FillRegister
from .utilities.filler_utils import rng, gen_answer_and_options
from .utilities.rounding import near_tie
function_registry = FillRegister()

def gen_rounded_number(a, b, decimals = 2):
    return round(rng.random()*b+a, decimals)

LAW_OF_DEMAND_PRICE_RANGES = {
    # domain: (low, width) of the uniform price draw
    'sports': (40, 200),
    'consumer_goods': (400, 1000),
    'medical': (40, 200),
    'finance': (4000, 20000),
}


@function_registry.register('law_of_demand')
def gen_law_of_demand_values(metadata):
    """
    Types price_increase / price_decrease ask for the new quantity;
    quantity_increase / quantity_decrease ask for the new price.

    Exactly one option lies on the side of the starting value that the law
    of demand predicts (price up -> smaller quantity, price down -> larger
    quantity, and conversely for the price asked about); the three
    distractors all lie strictly on the other side. The price and the
    quantity always both change.
    """
    typ = metadata['type']
    low, width = LAW_OF_DEMAND_PRICE_RANGES.get(metadata['domain'], (40, 200))

    def draw_price():
        return round(low + rng.random() * width, 2)

    while True:
        start_price, end_price = draw_price(), draw_price()
        start_quantity, end_quantity = int(rng.integers(1, 12)), int(rng.integers(1, 12))
        if abs(end_price - start_price) < 0.01 * start_price or start_quantity == end_quantity:
            continue
        price_up = end_price > start_price
        # enforce the law of demand on the pair
        if price_up != (end_quantity < start_quantity):
            continue
        if typ in ('price_increase', 'quantity_decrease') and not price_up:
            continue
        if typ in ('price_decrease', 'quantity_increase') and price_up:
            continue
        if typ == 'price_decrease' and start_quantity < 3:
            continue  # need three distinct smaller quantities (>= 0)
        break

    if typ in ('price_increase', 'price_decrease'):
        answer = end_quantity
        if typ == 'price_increase':
            # law-violating side: larger than the start quantity
            span = max(3, start_quantity - end_quantity + 2)
            pool = list(range(start_quantity + 1, start_quantity + span + 1))
        else:
            # law-violating side: smaller than the start quantity
            pool = list(range(0, start_quantity))
        distractors = [int(x) for x in rng.choice(pool, size=3, replace=False)]
    else:
        answer = end_price
        gap = max(abs(end_price - start_price), 0.05 * start_price)
        if typ == 'quantity_decrease':
            gap = min(gap, 0.9 * start_price)   # the distractors are lower prices, which must stay positive
        distractors = []
        while len(distractors) < 3:
            step = round(float(rng.uniform(0.05, 1.0)) * gap, 2)
            if typ == 'quantity_increase':
                cand = round(start_price + step, 2)   # price rose: violates the law
            else:
                cand = round(start_price - step, 2)   # price fell: violates the law
            if cand <= 0 or cand == start_price:
                continue
            # well apart (review: 923.55 next to 924.14); 3 fit in the 0.95 * gap range
            if all(abs(cand - d) >= max(0.01, 0.15 * gap) for d in distractors):
                distractors.append(cand)

    answer_mask, options = gen_answer_and_options(answer, options=[answer] + distractors)

    return {
        'start_price': start_price,
        'end_price': end_price,
        'start_quantity': start_quantity,
        'end_quantity': end_quantity,
        'correct_answer': answer_mask,
        'options_list': options
    }

PRICE_ELASTICITY_METHOD = ('Compute the percentage changes relative to the initial price and quantity '
                           '(not the midpoint formula).')


def arc_elasticity(q0, q1, p0, p1):
    """Midpoint (arc) elasticity: the other common convention; never offered as a distractor."""
    return ((q1 - q0) / ((q0 + q1) / 2)) / ((p1 - p0) / ((p0 + p1) / 2))


@function_registry.register('price_elasticity')
def gen_elasticity_values(metadata):
    """
    Price elasticity of demand by the simple percentage-change formula, relative to the initial values (the
    templates say so: template_fixes inserts PRICE_ELASTICITY_METHOD). Quantities are whole numbers (they count
    clients, units, ...). Prices change by at least 5% and the elasticity is between -0.1 and -10 in size, so the
    key never rounds to 0. Distractors are common mistakes (the inverse ratio, the slope dQ/dP, percentage changes
    relative to the final values, double, half), at least 0.05 and 10% from the key, from each other and from the
    midpoint-formula value (and its sign flip). In half the questions no option is a sign flip; in the other half the
    options are two +/- pairs, the key and one mistake each with its sign flipped, so no pair rule finds the key.
    """
    from .utilities.story_utils import pick_apart
    rising = 'price_increase' in metadata['type']
    while True:
        p_lo = round(float(rng.random()) * 95 + 5, 2)
        p_hi = round(float(rng.random()) * 95 + 5, 2)
        q_lo = int(rng.integers(5, 500))
        q_hi = int(rng.integers(5, 500))
        if not (p_lo < p_hi and q_lo < q_hi) or (p_hi - p_lo) / p_lo < 0.05 or round(p_lo * 100) % 10 == 0 \
                or round(p_hi * 100) % 10 == 0:
            continue
        if rising:
            start_price, end_price, start_quantity, end_quantity = p_lo, p_hi, q_hi, q_lo
        else:
            start_price, end_price, start_quantity, end_quantity = p_hi, p_lo, q_lo, q_hi
        elasticity = ((end_quantity - start_quantity) / start_quantity) / ((end_price - start_price) / start_price)
        if not 0.1 <= abs(elasticity) <= 10:
            continue
        arc = arc_elasticity(start_quantity, end_quantity, start_price, end_price)
        final_based = ((end_quantity - start_quantity) / end_quantity) / ((end_price - start_price) / end_price)
        slope = (end_quantity - start_quantity) / (end_price - start_price)
        if near_tie(elasticity):
            continue                  # exactly on a rounding tie (e.g. -0.125): the printed key would depend on the rule
        key = round(elasticity, 2)
        if abs(key - round(arc, 2)) < max(0.05, 0.1 * abs(key)):
            continue
        # Review, round 2: "-key" was the only sign-flipped twin among the options, so "the negative
        # member of the +/- pair" found the key. Now either no option is a sign flip (three same-sign mistakes),
        # or there are two +/- pairs: the key with its sign flip and a mistake with its sign flip.
        cands = [c for c in (1 / elasticity, slope, final_based, 2 * elasticity, elasticity / 2)
                 if abs(round(c, 2) - round(arc, 2)) >= max(0.05, 0.1 * abs(arc)) and round(c, 2) != 0
                 and abs(round(c, 2) + round(arc, 2)) >= max(0.05, 0.1 * abs(arc))]
        cands = [round(c, 2) for c in cands]
        if rng.random() < 0.5:
            dist = pick_apart(rng, key, cands, n=3, rel=0.1, abs_gap=0.05)
        else:
            one = pick_apart(rng, key, [c for c in cands if abs(abs(c) - abs(key)) >= max(0.05, 0.1 * abs(key))],
                             n=1, rel=0.1, abs_gap=0.05)
            dist = [-key, one[0], -one[0]] if one else None
        if dist:
            break
    options = [key] + dist
    answer_mask, options = gen_answer_and_options(key, options=options)
    return {
        'start_price': start_price,
        'end_price': end_price,
        'start_quantity': start_quantity,
        'end_quantity': end_quantity,
        'options_list': options,
        'correct_answer': answer_mask
    }

# income_elasticity: replaced by filler_tools/income_elasticity_tools.py (code templates), 5 October 2026.


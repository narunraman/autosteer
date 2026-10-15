from .utilities.production_utils import sample_prod_function,find_profit_max_CD,gen_answer_and_options,sample_agg_demand_function
from .fill_registry import FillRegister
from .utilities.filler_utils import rng
from .utilities.mistake_options import mistake_options
import math
from .utilities.aggregation_options import function_options, numeric_options, total_mistakes
function_registry = FillRegister()

# TODO: refactor two functions below (they are the same, just differet var names)

@function_registry.register('aggregation_of_consumer_demand')
def gen_agg_consumer_demand(metadata):
    # currently only does linear demand and the metadata type is "two_markets" so its asking for the total demand of two markets
    # need to return the following: num_consumers, num_a, num_b, demand_function_a, demand_function_b, price, options, correct_answer
    def gen_values():
        num_a = int(rng.integers(1,1000))
        num_b = int(rng.integers(1,1000))
        num_consumers = num_a + num_b
        a, coef_a = round(rng.random()*100,2), round(rng.random()*10,2)
        b, coef_b = round(rng.random()*100,2), round(rng.random()*10,2)

        demand_function_a = f"{a} - {coef_a}P"
        demand_function_b = f"{b} - {coef_b}P"

        price = round(rng.random()*10,2)

        return num_consumers, num_a, num_b, a, coef_a,  b, coef_b, demand_function_a, demand_function_b, price
    
    # Redraw until each segment's individual demand is strictly positive at
    # the asked price, so the aggregate is simply the sum of the segments
    # (no truncation of negative demand at zero is needed).
    while True:
        num_consumers, num_a, num_b, a, coef_a,  b, coef_b, demand_function_a, demand_function_b, price = gen_values()
        if a - coef_a*price > 0 and b - coef_b*price > 0 and min(coef_a, coef_b) >= 0.05 and min(num_a, num_b) >= 10 and num_a != num_b:
            break
    # Options (review, October 2026): every distractor is an aggregation mistake on the key's scale and
    # the key's rank among the options is random (utilities/aggregation_options.py).
    if metadata['type'] == 'demand_function':
        while True:
            got = function_options(rng, num_a, num_b, a, coef_a, b, coef_b, -1)
            if got is not None:
                break
            num_consumers, num_a, num_b, a, coef_a, b, coef_b, demand_function_a, demand_function_b, price = gen_values()
        options, answer_mask = got
        return {
            "num_consumers": num_consumers,
            "num_a": num_a,
            "num_b": num_b,
            "demand_function_a": demand_function_a,
            "demand_function_b": demand_function_b,
            "options_list": [options],
            "correct_answer": [answer_mask]
        }

    # Numerical version (quantity demanded):
    while True:
        key = num_a*(a - coef_a*price) + num_b*(b - coef_b*price)
        got = numeric_options(rng, key, total_mistakes(num_a, num_b, a, coef_a, b, coef_b, -1, price))
        if got is not None:
            break
        num_consumers, num_a, num_b, a, coef_a, b, coef_b, demand_function_a, demand_function_b, price = gen_values()
    options, answer_mask = got
    return {
        "num_consumers": num_consumers,
        "num_a": num_a,
        "num_b": num_b,
        "demand_function_a": demand_function_a,
        "demand_function_b": demand_function_b,
        "price": price,
        "options_list": [options],
        "correct_answer": [answer_mask]
    }


@function_registry.register('aggregation_of_labor_demand')
def gen_agg_labor_demand(metadata):
    return _factor_aggregation(metadata, 'num_firms')


def _factor_aggregation(metadata, total_key):
    """
    aggregation_of_labor_demand / aggregation_of_capital_demand (Review, October 2026). Two groups of n_a and
    n_b firms (markets) with linear demands a - cP; the aggregate is (n_a a + n_b b) - (n_a c_a + n_b c_b)P where both
    demands are nonnegative. Intercepts are at least 1 and slopes at least 0.1, and every question states the price
    range (demand_function type: 0 <= P <= p_max, the lower choke price, rounded down; the template fix says so) or
    asks at a price below both choke prices (numerical type), so no group's demand is negative.
    Distractors are aggregation mistakes, three chosen at random: the two demands added unweighted, the group sizes
    swapped, both demands weighted by the total count, the per-firm average (the aggregate divided by the total
    count), one group left out (one of the two ways, else the key is the sum of two options), and (numerical) the price term left out. Some are above and some below
    the key.
    """
    while True:
        num_a, num_b = int(rng.integers(1, 1000)), int(rng.integers(1, 1000))
        a, coef_a = round(float(rng.uniform(1, 100)), 2), round(float(rng.uniform(0.1, 10)), 2)
        b, coef_b = round(float(rng.uniform(1, 100)), 2), round(float(rng.uniform(0.1, 10)), 2)
        choke = min(a / coef_a, b / coef_b)
        if choke < 1:
            continue
        N = num_a + num_b
        price = round(float(rng.uniform(0.5, 0.9 * choke)), 2)
        p_max = math.floor(choke * 100) / 100
        common = {total_key: N, 'num_a': num_a, 'num_b': num_b,
                  'demand_function_a': f"{a} - {coef_a}P", 'demand_function_b': f"{b} - {coef_b}P"}
        if metadata['type'] == 'demand_function':
            def fn(i, s):
                return (round(i, 2), round(s, 2))
            key = fn(num_a * a + num_b * b, num_a * coef_a + num_b * coef_b)
            cands = [fn(a + b, coef_a + coef_b),
                     fn(num_b * a + num_a * b, num_b * coef_a + num_a * coef_b),
                     fn(N * (a + b), N * (coef_a + coef_b)),
                     fn(key[0] / N, key[1] / N),
                     # one group left out; only one of the two, or the key would be the sum of two options
                     fn(num_a * a, num_a * coef_a) if rng.random() < 0.5 else fn(num_b * b, num_b * coef_b)]
            chosen = []
            for c in [cands[int(t)] for t in rng.permutation(len(cands))]:
                if all(abs(c[0] - o[0]) > max(0.015, 0.03 * abs(o[0])) or abs(c[1] - o[1]) > max(0.015, 0.03 * abs(o[1]))
                       for o in [key] + chosen):
                    chosen.append(c)
                if len(chosen) == 3:
                    break
            if len(chosen) < 3:
                continue
            texts = [f"{i} - {s}P" for i, s in [key] + chosen]
            correct = texts[0]
            rng.shuffle(texts)
            return {**common, 'p_max': f"{p_max:.2f}", 'options_list': [texts],
                    'correct_answer': [[t == correct for t in texts]]}
        qa, qb = a - coef_a * price, b - coef_b * price
        key = num_a * qa + num_b * qb
        res = mistake_options(key, [qa + qb, num_b * qa + num_a * qb, N * (qa + qb), key / N, num_a * a + num_b * b,
                               num_a * qa if rng.random() < 0.5 else num_b * qb],
                              fill=False, rel=0.03)
        if res is None:
            continue
        mask, options = res
        return {**common, 'price': price, 'options_list': options, 'correct_answer': mask}


@function_registry.register('aggregate_demand')
def gen_aggregate_demand(metadata):
    if 'simple' in metadata['type']:
        family = 'quadratic'
        s_func = sample_agg_demand_function(family)
        price = round(rng.random()*10,2)
        num_firms = int(rng.integers(1,20))
        labor = price**2*s_func['a'] + price*s_func['b'] + s_func['c']
        total_labor = labor*num_firms
        answer_mask,options = gen_answer_and_options(total_labor)
        return {
            "s_func": s_func['function_string'],
            "price": price,
            "num_firms": num_firms,
            "options_list":options,
            "correct_answer": answer_mask
        }
    else:
        family = 'Cobb_Douglas'
        no_answer = True
        while no_answer:
            prod_func = sample_prod_function(family)
            price = round(rng.random()*10,2)
            num_firms = int(rng.integers(1,20))
            cost = round(rng.random()*10,2)
            capital = round(rng.random()*10,2) 
            opt_labor = find_profit_max_CD(prod_func,cost,price,capital)
            if opt_labor:
                no_answer = False
        total_labor = opt_labor*num_firms
        answer_mask,options = gen_answer_and_options(total_labor)
        if "capital" in metadata['type']:
            p_func = f"{capital**prod_func['K']}K^{prod_func['L']}"
        else:
            p_func = f"{capital**prod_func['K']}L^{prod_func['L']}"
        return {
            "p_func": p_func,
            "cost": cost,
            "price": price,
            "num_firms": num_firms,
            "options_list":options,
            "correct_answer": answer_mask
        }

@function_registry.register('aggregation_of_capital_demand')
def gen_agg_capital_demand(metadata):
    # Two groups of markets with linear capital demands; see _factor_aggregation.
    if metadata['type'] not in ('demand_function', 'numerical'):
        return None
    return _factor_aggregation(metadata, 'num_markets')

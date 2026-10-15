from .fill_registry import FillRegister
from .utilities.filler_utils import rng
from .utilities.aggregation_options import function_options, numeric_options, total_mistakes
function_registry = FillRegister()

# TODO: refactor two functions below (they are the same, just differet var names)

@function_registry.register('aggregation_of_offer_curve')
def gen_agg_offer_curve(metadata):
    # Generates aggregated offer curve for a good across two markets.
    def gen_values():
        num_a = int(rng.integers(1, 1000))
        num_b = int(rng.integers(1, 1000))
        num_markets = num_a + num_b
        a, coef_a = round(rng.random() * 100, 2), round(rng.random() * 10, 2)
        b, coef_b = round(rng.random() * 100, 2), round(rng.random() * 10, 2)
        
        offer_curve_a = f"{a} + {coef_a}P"
        offer_curve_b = f"{b} + {coef_b}P"
        
        price = round(rng.random() * 10, 2)
        
        return num_markets, num_a, num_b, a, coef_a, b, coef_b, offer_curve_a, offer_curve_b, price
    
    # Options (review, October 2026): every distractor is an aggregation mistake on the key's scale and
    # the key's rank among the options is random (utilities/aggregation_options.py); the old function distractors
    # were random curves on another scale, and the numerical key was always the largest option.
    while True:
        num_markets, num_a, num_b, a, coef_a, b, coef_b, offer_curve_a, offer_curve_b, price = gen_values()
        if min(a, b, coef_a, coef_b) < 0.05 or min(num_a, num_b) < 10 or num_a == num_b:
            continue
        if metadata['type'] == 'offer_curve':
            got = function_options(rng, num_a, num_b, a, coef_a, b, coef_b, +1)
        elif metadata['type'] == 'numerical':
            key = num_a * (a + coef_a * price) + num_b * (b + coef_b * price)
            got = numeric_options(rng, key, total_mistakes(num_a, num_b, a, coef_a, b, coef_b, +1, price))
        else:
            return None
        if got is not None:
            break
    options, answer_mask = got
    out = {
        "num_markets": num_markets,
        "num_a": num_a,
        "num_b": num_b,
        "offer_curve_a": offer_curve_a,
        "offer_curve_b": offer_curve_b,
        "options_list": [options],
        "correct_answer": [answer_mask]
    }
    if metadata['type'] == 'numerical':
        out["price"] = price
    return out

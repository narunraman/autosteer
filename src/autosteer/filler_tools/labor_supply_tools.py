# This file contains functions that generate values for labor supply related questions

# External packages

# Local packages
from .fill_registry import FillRegister
from .utilities.filler_utils import rng, gen_answer_and_options
from .utilities.rank_pick import pick_ranked
function_registry = FillRegister()


# labor_supply_elasticity: replaced by filler_tools/labor_supply_elasticity_tools.py, code templates (review,
# October 2026).


BUDGET_TEXT = ('wh+v where w is the hourly wage in dollars, h is the number of hours worked per day, and v is non-labor '
               'income in dollars per day, with 24 hours per day to divide between work h and leisure L = 24 - h')
# the function type's key is the interior solution; below the wage b v / (24 a) the worker does not work at all
INTERIOR_TEXT = ', and the wage is high enough that the worker chooses to work some hours (h > 0)'


@function_registry.register('deriving_labor_supply')
def gen_derive_labor_supply_values(metadata):
    """
    max C^a L^b s.t. C = w h + v, L = 24 - h (w hourly, v per day; stated in the budget text). FOC:
    a w (24 - h) = b (w h + v), so
        h*(w) = (24 a w - b v) / ((a + b) w),
    for w > b v / (24 a); below that the worker does not work (h = 0). The function type's budget text says that the
    wage is in the interior range (review, October 2026: the keyed function went negative for low wages).
    (Narun's default: the old denominator was b - a.) Draws with h* outside (0.5, 23.5) are redrawn. Computed from
    the displayed values.
    Hours-type distractors are mistakes between 0 and 24 hours (an option above the 24-hour day was infeasible on
    its face): |b - a| as the denominator, non-labor income ignored, the leisure share's hours net of v / w, the sign
    of v flipped, v counted twice, v not weighted by b / (a + b), and the optimal leisure 24 - h*; a random number
    of them below the key.
    """
    r2 = lambda x: round(float(x), 2)
    if metadata['type'] in ('cobb-douglas_hours', 'hours'):
        while True:
            a, b = (int(x) for x in rng.integers(1, 10, size=2))
            wage, v = r2(rng.uniform(1, 100)), r2(rng.uniform(0, 100))
            h = (24 * a * wage - b * v) / ((a + b) * wage)
            if a == b or not 0.5 < h < 23.5:
                continue
            cands = [(24 * a * wage - b * v) / (abs(b - a) * wage), 24 * a / (a + b), 24 * b / (a + b) - v / wage,
                     (24 * a * wage + b * v) / ((a + b) * wage), 24 - h,
                     (24 * a * wage - 2 * b * v) / ((a + b) * wage), 24 * a / (a + b) - v / wage]
            dist = pick_ranked(rng, h, cands, rel=0.05, abs_gap=0.1, lo=0.1, hi=23.9, shown=lambda x: r2(x))
            if dist is not None:
                break
        options = [r2(x) for x in [h] + dist]
        return {
            'wage': wage,
            'non_labor_income': v,
            'budget': BUDGET_TEXT,
            'utility_func': f"U = C^{a}L^{b}",
            'options_list': [options],
            'correct_answer': [[1, 0, 0, 0]]
        }
    elif metadata['type'] in ('cobb-douglas_function', 'function'):
        # Options (A w - B)/(D w), written by their (A, B, D). Distractors are mistakes: |b - a| as the denominator,
        # a and b swapped in the numerator (both terms, or only the 24 term, or only the v term), v not weighted by
        # b, the 24-hour endowment dropped. All have the key's minus sign (a sign flip would be a never-key cue). Three are chosen so that no two options
        # are within 3% in all of A, B and D and the key does not share more parts with the other options than some
        # distractor does, and the key is not the per-part majority (a key that is the "hub" of the options
        # gives itself away).
        def shared(x, others):
            return sum(x[k] == o[k] for o in others for k in range(3))
        while True:
            a, b = r2(rng.uniform(0.1, 20)), r2(rng.uniform(0.1, 20))
            v = r2(rng.uniform(1, 100))
            if abs(a - b) <= 0.5:
                continue
            key = (r2(24 * a), r2(b * v), r2(a + b))
            pool = [(key[0], key[1], r2(abs(b - a))), (r2(24 * b), r2(a * v), key[2]), (key[0], r2(v), key[2]),
                    (r2(a), key[1], key[2]), (r2(24 * b), key[1], key[2]), (key[0], r2(a * v), key[2])]
            pick = [pool[int(k)] for k in rng.permutation(len(pool))[:3]]
            opts = [key] + pick
            near = any(all(abs(x[k] - y[k]) <= 0.03 * max(abs(x[k]), abs(y[k])) for k in range(3))
                       for n_, x in enumerate(opts) for y in opts[n_ + 1:])
            if near or shared(key, pick) > max(shared(d, [o for o in opts if o is not d]) for d in pick):
                continue
            # nor is the key the per-part majority (most common A, B and D among the options)
            if all(sum(o[k] == key[k] for o in opts) > max(sum(o[k] == x[k] for o in opts) for x in opts
                                                             if x[k] != key[k]) if any(x[k] != key[k] for x in opts)
                   else True for k in range(3)):
                continue
            break
        fmt = lambda f: f"({f[0]}w - {f[1]})/({f[2]}w)"
        return {
            'non_labor_income': v,
            'budget': BUDGET_TEXT + INTERIOR_TEXT,
            'utility_func': f"U = C^({a})L^({b})",
            'options_list': [[fmt(f) for f in opts]],
            'correct_answer': [[1, 0, 0, 0]]
        }
    raise ValueError(f"Unknown deriving_labor_supply type {metadata['type']!r}")


# mrs_labor_supply: replaced by filler_tools/mrs_labor_supply_tools.py, code templates, 7 October 2026.
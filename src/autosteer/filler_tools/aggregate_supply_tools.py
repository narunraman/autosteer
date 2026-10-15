from .fill_registry import FillRegister
from .utilities.filler_utils import rng
from .utilities.rank_pick import pick_ranked
from .utilities.story_utils import num
function_registry = FillRegister()


@function_registry.register('aggregation_of_labor_supply')
def gen_agg_labor_supply(metadata):
    """
    Two groups of num_a and num_b members, each member facing labor supply a + c_a P (group a) or b + c_b P (group b).
    The aggregate is (num_a a + num_b b) + (num_a c_a + num_b c_b) P; the numerical type evaluates it at the wage P.

    Review, October 2026: the distractors were the key's components times random factors below 1 (and a random
    function), so the key was always the largest option. They are now mistakes, with a random number of them below
    the key (numerical type): the group sizes swapped, the curves added without weights, every member on group a's
    curve or on group b's curve, the intercepts or the slopes left unweighted, the average curve times one group's
    size, both curves counted for every member (in both
    parts or in one). Function type: the same mistakes applied to intercept and slope, except those that keep the key's intercept or slope; an option differs from the key and the
    others by at least 5% in intercept or slope. Groups have at least 5 members.
    """
    r2 = lambda v: round(float(v), 2)
    for _ in range(100000):
        num_a, num_b = int(rng.integers(5, 1000)), int(rng.integers(5, 1000))
        a, coef_a = r2(rng.uniform(1, 100)), r2(rng.uniform(0.5, 10))
        b, coef_b = r2(rng.uniform(1, 100)), r2(rng.uniform(0.5, 10))
        price = r2(rng.uniform(1, 10))
        n = num_a + num_b
        key = (num_a * a + num_b * b, num_a * coef_a + num_b * coef_b)
        mistakes = [
            (num_b * a + num_a * b, num_b * coef_a + num_a * coef_b),      # group sizes swapped
            (a + b, coef_a + coef_b),                                      # no weights
            (n * a, n * coef_a), (n * b, n * coef_b),                      # everyone on one curve
            (a + b, key[1]), (key[0], coef_a + coef_b),                    # one part unweighted
            (num_a * (a + b) / 2, num_a * (coef_a + coef_b) / 2),          # average curve times one group
            (num_b * (a + b) / 2, num_b * (coef_a + coef_b) / 2),
            (n * (a + b), n * (coef_a + coef_b)),                          # both curves for every member
            (n * (a + b), key[1]), (key[0], n * (coef_a + coef_b)),        # one part counted for every member
        ]
        if metadata['type'] == 'numerical':
            val = lambda f: f[0] + f[1] * price
            dist = pick_ranked(rng, val(key), [val(m) for m in mistakes], rel=0.05, abs_gap=0.5, shown=num)
            if dist is None:
                continue
            options = [num(x) for x in [val(key)] + dist]
        else:
            fmt = lambda f: f"{num(f[0])} + {num(f[1])}P"
            def apart(f, g):
                return abs(f[0] - g[0]) > 0.05 * max(abs(f[0]), abs(g[0])) or \
                    abs(f[1] - g[1]) > 0.05 * max(abs(f[1]), abs(g[1]))
            chosen = []
            # no option shares its intercept or slope with the key (that would single the key out)
            pool = [m for m in mistakes if m[0] != key[0] and m[1] != key[1]]
            for i in rng.permutation(len(pool)):
                m = pool[int(i)]
                if all(apart(m, x) for x in [key] + chosen):
                    chosen.append(m)
                if len(chosen) == 3:
                    break
            if len(chosen) < 3:
                continue
            options = [fmt(f) for f in [key] + chosen]
        if len(set(options)) < 4:
            continue
        out = {
            "num_firms": n,
            "num_a": num_a,
            "num_b": num_b,
            "supply_function_a": f"{num(a)} + {num(coef_a)}P",
            "supply_function_b": f"{num(b)} + {num(coef_b)}P",
            "options_list": [options],
            "correct_answer": [[1, 0, 0, 0]],
        }
        if metadata['type'] == 'numerical':
            out["price"] = price
        return out
    raise RuntimeError('aggregation_of_labor_supply: no draw')

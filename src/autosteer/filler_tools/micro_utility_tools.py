import itertools
import re

from .fill_registry import FillRegister
from .utilities.filler_utils import rng, gen_answer_and_options
from .utilities.micro_utility_utils import sample_utility_function
function_registry = FillRegister()


def no_feature_majority(key: tuple, chosen: list) -> bool:
    """True if, for every feature, the key's value is not the unique most common value among key + chosen: a
    reader who picks the option whose features agree with most others (the usual weakness of options built as
    one-mistake variants of the key) cannot find the key that way."""
    opts = [key] + list(chosen)
    for f in range(len(key)):
        counts = {}
        for o in opts:
            counts[o[f]] = counts.get(o[f], 0) + 1
        if any(v >= counts[key[f]] for k, v in counts.items() if k != key[f]):
            continue
        return False
    return True


def balanced_distractors(key: tuple, candidates: list, n: int = 3) -> list:
    """n candidates chosen at random among the sets that pass no_feature_majority."""
    sets = [c for c in itertools.combinations(candidates, n) if no_feature_majority(key, c)]
    return list(sets[int(rng.integers(0, len(sets)))])


def _cd_term(coef, ex_x, ex_y) -> str:
    c = '' if coef is None else f"{coef}"
    return f"{c}x^{ex_x}y^{ex_y}"


def cd_marginal_utility_options(alpha: float, beta: float) -> tuple[str, list[str]]:
    """Key "(MU_x, MU_y)" for U = x^alpha y^beta and three distractors, each a natural differentiation mistake built
    from the same exponents (review, October 2026: distractors with other exponents left the key as
    the only option whose coefficients matched the utility). Mistakes: the coefficient left out; the exponent lowered
    on the other good or on neither; the pair swapped (MU_y, MU_x); combinations of these. Several combinations print
    the same (without coefficients, lowering the other good's exponent is the swapped pair), so options are compared
    by what they show: coefficients written or not, which good's exponent is lowered in the first and in the second
    entry, and the first entry's coefficient. The three are drawn so that no shown feature of the key is the majority
    (see no_feature_majority)."""
    r = lambda v: round(v, 2)
    ex = {'x': alpha, 'y': beta}

    def comp(var, coef_on, lower):
        other = 'y' if var == 'x' else 'x'
        e = dict(ex)
        low = {'own': var, 'other': other}.get(lower)
        if low:
            e[low] = r(e[low] - 1)
        text = _cd_term(ex[var] if coef_on else None, e['x'], e['y'])
        return text, low, (ex[var] if coef_on else None)

    by_text = {}
    for coef_on, lower, swapped in itertools.product((True, False), ('own', 'other', 'none'), (False, True)):
        pair = [comp('x', coef_on, lower), comp('y', coef_on, lower)]
        if swapped:
            pair = pair[::-1]
        text = f"({pair[0][0]}, {pair[1][0]})"
        shown = (coef_on, pair[0][1], pair[1][1], pair[0][2])
        by_text.setdefault(text, shown)
    key_text = f"({comp('x', True, 'own')[0]}, {comp('y', True, 'own')[0]})"
    key = by_text.pop(key_text)
    # U itself in both entries (no coefficient, nothing lowered) is not offered
    cands = [(t, f) for t, f in by_text.items() if not (f[0] is False and f[1] is None)]
    feats = [f for _, f in cands]
    chosen = balanced_distractors(key, feats)
    texts = [t for t, f in cands if f in chosen]
    return key_text, [key_text] + texts


def cd_mrs_options(alpha: float, beta: float) -> tuple[str, list[str]]:
    """Key "((alpha y)/(beta x), (beta x)/(alpha y))" = (MU_x/MU_y, MU_y/MU_x) for U = x^alpha y^beta. The options
    are the 2 x 2 grid {alpha/beta, beta/alpha} x {y/x, x/y} (review, October 2026: distractors with
    other exponents left the key as the only option repeating the utility's exponents). Every option is built from
    the shown exponents and each feature occurs in two options."""
    def text(c1, c2, v1, v2):
        return f"(({c1}{v1})/({c2}{v2}), ({c2}{v2})/({c1}{v1}))"
    key = text(alpha, beta, 'y', 'x')
    return key, [key, text(beta, alpha, 'y', 'x'), text(alpha, beta, 'x', 'y'), text(beta, alpha, 'x', 'y')]


@function_registry.register("marginal_utility")
def gen_marginal_utility_values(metadata):

    def gen_values():
        if metadata['type'] == 'cobb_douglas':
            # round first so the shown exponents and alpha - 1 agree
            alpha, beta = (round(float(v), 2) for v in rng.uniform(0.1, 0.9, 2))
            utility_function = f"U(x, y) = x^{alpha}y^{beta}"
            mu_x = f"{alpha}x^{round(alpha-1, 2)}y^{beta}"
            mu_y = f"{beta}x^{alpha}y^{round(beta-1, 2)}"

        elif metadata['type'] == 'linear':
            alpha, beta = rng.uniform(0.2, 20, 2)
            utility_function = f"U(x, y) = {alpha}x + {beta}y"
            mu_x = f"{alpha}"
            mu_y = f"{beta}"

        # Leontief type dropped (Narun's default): min() has no derivative at the kink.

        elif metadata['type'] == 'non-linear':
            # create an arbitrary non-linear utility function
            x_coef, y_coef, mix_coef = (round(float(v), 2) for v in rng.uniform(0.1, 20, 3))
            alpha, beta = (int(v) for v in rng.integers(1, 10, 2))
            utility_function = f"U(x, y) = {x_coef}x^{alpha} + {mix_coef}xy + {y_coef}y^{beta}"
            # d/dx c x^1 = c (it was shown as c x)
            mu_x = (f"{round(alpha*x_coef, 2)}x^{alpha-1} + {mix_coef}y" if alpha > 2 else f"{round(alpha*x_coef, 2)}x + {mix_coef}y") if alpha > 1 else f"{x_coef} + {mix_coef}y"
            mu_y = (f"{round(beta*y_coef, 2)}y^{beta-1} + {mix_coef}x" if beta > 2 else f"{round(beta*y_coef, 2)}y + {mix_coef}x") if beta > 1 else f"{y_coef} + {mix_coef}x"

        else:
            raise ValueError("Invalid utility function type")

        return alpha, beta, utility_function, mu_x, mu_y
    
    alpha, beta, utility_function, mu_x, mu_y = gen_values()
    if metadata['type'] == 'cobb_douglas':
        while abs(alpha - beta) < 0.05:   # the options differ by which exponent goes where
            alpha, beta, utility_function, mu_x, mu_y = gen_values()
        correct_answer, options = cd_marginal_utility_options(alpha, beta)
    else:
        correct_answer = f"({mu_x}, {mu_y})"
        options = [correct_answer]
        for i in range(3):
            _, _, _, mu_x, mu_y = gen_values()
            options.append(f"({mu_x}, {mu_y})")

    answer_mask, options_list = gen_answer_and_options(correct_answer, 4, options)
    
    
    return {
        # 'alpha': alpha,
        # 'beta': beta,
        'utility_function': utility_function,
        'correct_answer': answer_mask,
        'options_list': options_list
    }


@function_registry.register("mrs_utility")
def gen_mrs_values(metadata):

    "(MU_x/MU_y, MU_y/MU_x)"

    # x = rng.uniform(1, 100)
    # y = rng.uniform(1, 100)
    if metadata['type'] == 'cobb_douglas':
        while True:
            alpha, beta = (round(float(v), 2) for v in rng.uniform(0.1, 0.9, 2))
            if abs(alpha - beta) >= 0.05:   # the options differ by which exponent goes where
                break
        utility_function = f"U(x, y) = x^{alpha}y^{beta}"
        mrs, options = cd_mrs_options(alpha, beta)
        answer_mask, options_list = gen_answer_and_options(mrs, 4, options)
        return {
            'utility_function': utility_function,
            'correct_answer': answer_mask,
            'options_list': options_list
        }
        #((alpha/beta) * (y/x), (beta/alpha) * (x/y))

    elif metadata['type'] == 'linear':
        alpha, beta = rng.uniform(0.2, 20, 2)
        utility_function = f"U(x, y) = {alpha}x + {beta}y"
        mrs = (alpha/beta, beta/alpha)

    # Leontief type dropped (Narun's default): MRS undefined at the kink.

    elif metadata['type'] == 'non-linear':
        # create an arbitrary non-linear utility function
        def gen_nl_values():
            x_coef, y_coef, mix_coef = (round(float(v), 2) for v in rng.uniform(0.1, 20, 3))
            alpha, beta = (int(v) for v in rng.integers(1, 10, 2))
            utility_function = f"U(x, y) = {x_coef}x^{alpha} + {mix_coef}xy + {y_coef}y^{beta}"

            # Take the derivative of the utility function with respect to x and y (d/dx c x^1 = c)
            mu_x = (f"{round(alpha*x_coef, 2)}x^{alpha - 1} + {mix_coef}y" if alpha > 2 else f"{round(alpha*x_coef, 2)}x + {mix_coef}y") if alpha > 1 else f"{x_coef} + {mix_coef}y"
            mu_y = (f"{round(beta*y_coef, 2)}y^{beta - 1} + {mix_coef}x" if beta > 2 else f"{round(beta*y_coef, 2)}y + {mix_coef}x") if beta > 1 else f"{y_coef} + {mix_coef}x"

            mrs = f"(({mu_x})/({mu_y}), ({mu_y})/({mu_x}))"
            return utility_function, mrs
        
        utility_function, mrs = gen_nl_values()
        options = [mrs] + [gen_nl_values()[1] for _ in range(3)]
        answer_mask, options_list = gen_answer_and_options(mrs, 4, options)
        return {
            'utility_function': utility_function,
            'correct_answer': answer_mask,
            'options_list': options_list
        }



        
    else:
        raise ValueError("Invalid utility function type")
    
    answer_mask, options_list = gen_answer_and_options(mrs, 4)

    return {
        'utility_function': utility_function,
        'correct_answer': answer_mask,
        'options_list': options_list
    }

# @function_registry.register("mrs_tangency")
# def gen_mrs_tangency_values(metadata):


@function_registry.register('diminishing_marginal_utility')
def gen_diminishing_marginal_utility(metadata):
    """
    Types {x_fixed, y_fixed}_{linear, cobb_douglas, quadratic}: one good is
    held at `fixed`, the other rises from `start` to `end`; the answer
    describes the marginal utility of the varied good (x_fixed varies y).
    Mirrors diminishing_marginal_product: Cobb-Douglas exponent of the varied
    good in (0, 1), quadratic with negative square terms (MU positive over the
    range) -> "decreases from MU(start) to MU(end)"; linear -> "remains
    constant at <coefficient>". The old filler used the fixed good's
    coefficients for x_fixed types. Computed from the displayed values.
    """
    typ = metadata['type']
    var = 'y' if typ.startswith('x_fixed') else 'x'
    other = 'x' if var == 'y' else 'y'
    r2 = lambda v: round(float(v), 2)
    while True:
        start = int(rng.integers(1, 50))
        end = start + int(rng.integers(1, 10))
        fixed = int(rng.integers(1, 50))
        if 'cobb_douglas' in typ:
            ex = {var: r2(rng.uniform(0.1, 0.95)), other: r2(rng.uniform(0.1, 2))}
            fn = f"x^{ex['x']}y^{ex['y']}"
            mu = lambda v: ex[var] * v ** (ex[var] - 1) * fixed ** ex[other]
        elif 'quadratic' in typ:
            lin = {'x': r2(rng.uniform(1, 10)), 'y': r2(rng.uniform(1, 10))}
            sq = {'x': -r2(rng.uniform(0.01, 1)), 'y': -r2(rng.uniform(0.01, 1))}
            cross = r2(rng.uniform(0, 10))
            fn = (f"{lin['x']}x + {lin['y']}y - {abs(sq['x'])}x^2 - {abs(sq['y'])}y^2"
                  + (f" + {cross}xy" if cross else ''))
            mu = lambda v: lin[var] + 2 * sq[var] * v + cross * fixed
        else:
            coef = {'x': r2(rng.uniform(0.5, 10)), 'y': r2(rng.uniform(0.5, 10))}
            fn = f"{coef['x']}x + {coef['y']}y"
            mu = lambda v: coef[var]
        m0, m1 = r2(mu(start)), r2(mu(end))
        if 'linear' in typ or (m1 >= 0.01 and m0 - m1 >= 0.01):
            break

    up = r2(m0 + rng.uniform(0.1, 1) * max(m0, 1.0))
    increases = f"The marginal utility of {var} increases from {m0} to {up}"
    constant = f"The marginal utility of {var} remains constant at {m0}"
    if 'linear' in typ:
        decreases = f"The marginal utility of {var} decreases from {m0} to {r2(m0 * rng.uniform(0.1, 0.9))}"
        answer = constant
    else:
        decreases = f"The marginal utility of {var} decreases from {m0} to {m1}"
        answer = decreases
    answer_mask, options = gen_answer_and_options(answer, num_options=3, options=[increases, constant, decreases])
    return {
        "utility_func": fn,
        "start": start,
        "end": end,
        "fixed": fixed,
        "options_list": options,
        "correct_answer": answer_mask
    }


# ----------------------------------------------------------------------------------------------------------------
# micro_utility: STEER 1.2.a "Optimize over a Discrete Set"
# ----------------------------------------------------------------------------------------------------------------
# A utility function over goods x and y and a short list of bundles; the key is the bundle with the highest
# utility, computed here, and the distractors are the other listed bundles. Families: Cobb-Douglas x^a * y^b,
# linear a x + b y, perfect complements min(a x, b y). The best bundle beats the runner-up by at least 5%, and it
# never has at least as much of both goods as every other bundle (so the list cannot be ranked without the
# function). Templates that speak of two bundles ("two bundles", "both options", ...) get two, the others four.
TWO_BUNDLES = re.compile(r'\btwo\b(?:\s+[\w-]+){0,2}?\s+(?:bundles|options|combinations|choices|alternatives|packages|'
                         r'sets|baskets|selections|deals|offers|plans)\b|\bboth\s+(?:bundles|options|combinations|choices)\b',
                         re.I)

DISCRETE_FAMILIES = ('cobb_douglas', 'linear', 'min')


def discrete_utility(family: str, a: float, b: float, x: int, y: int) -> float:
    if family == 'cobb_douglas':
        return x ** a * y ** b
    if family == 'linear':
        return a * x + b * y
    if family == 'min':
        return min(a * x, b * y)
    raise ValueError(family)


def discrete_utility_text(family: str, a, b) -> str:
    if family == 'cobb_douglas':
        return f'x^{a} * y^{b}'
    ca, cb = ('' if c == 1 else c for c in (a, b))
    if family == 'linear':
        return f'{ca}x + {cb}y'
    return f'min({ca}x, {cb}y)'


def bundle_text(x: int, y: int) -> str:
    return f'x = {x}, y = {y}'


@function_registry.register('micro_utility')
def gen_micro_utility(metadata):
    text = metadata.get('question_text', '')
    n = 2 if TWO_BUNDLES.search(text) else 4
    family = DISCRETE_FAMILIES[int(rng.integers(0, len(DISCRETE_FAMILIES)))]
    while True:
        if family == 'cobb_douglas':
            a, b = (round(float(v), 1) for v in rng.uniform(0.15, 0.95, 2))
        else:
            a, b = (int(v) for v in rng.integers(1, 10, 2))
        bundles = set()
        while len(bundles) < n:
            bundles.add((int(rng.integers(1, 21)), int(rng.integers(1, 21))))
        bundles = sorted(bundles)
        utils = [discrete_utility(family, a, b, x, y) for x, y in bundles]
        order = sorted(range(n), key=lambda i: -utils[i])
        best, second = order[0], order[1]
        if utils[best] < 1.05 * utils[second]:
            continue
        bx, by = bundles[best]
        if all(bx >= x and by >= y for x, y in bundles):
            continue
        break
    func = discrete_utility_text(family, a, b)
    # "u(x, y) = {utility_func}" or "u(x, y) ... as: {utility_func}" in the template: give the bare expression
    named = re.search(r'=\s*\{utility_func(?:tion)?\}', text) or re.search(r'\b[uU]\s*\(\s*x\s*,\s*y\s*\)', text)
    shown = func if named else f'U(x, y) = {func}'
    options = [bundle_text(x, y) for x, y in bundles]
    rng.shuffle(options)
    mask = [int(o == bundle_text(bx, by)) for o in options]
    return {
        'utility_func': shown,
        'utility_function': shown,
        'options_list': [options],
        'correct_answer': [mask],
    }

import numpy as np
from .utilities.filler_utils import rng
from .utilities.filler_utils import gen_answer_and_options
from .fill_registry import FillRegister
function_registry = FillRegister()

# --------------------------------------------------------------------------------------------------------------
# exponents and linear_system_eq (STEER-ME Foundations), rebuilt 5 October 2026 with code templates
# --------------------------------------------------------------------------------------------------------------
# The old fillers (no templates) printed unrounded floats and drew distractors as random multiples of the answer.
#
# exponents (module Derivatives and Homotheticity): base^exponent with a base of 2 decimals in [1.1, 9.99].
#   integer: whole exponent 2..6; real_value: an exponent of 2 decimals in [0.25, 3.5] that is not a whole number.
#   The key is the power rounded to 2 decimals (at most 100,000). Distractors: base times exponent; one power too
#   few or too many (base^(e-1), base^(e+1)); the exponent as the root (base^(1/e)); exponent^base. Three at least
#   5% apart from the key and from each other.
# linear_system_eq (module Systems of Equations): two equations in x, y or three in x, y, z with whole coefficients
#   in -9..9 (at most one zero, none with two equations; |det| >= 2) and a whole-number solution in -12..12 with
#   distinct values. Options are whole-
#   number assignments: the solution; the values of x and y swapped (or the three values rotated); a point that
#   satisfies every equation but the last (the solution moved along the line where the other equations hold); the
#   solution with every sign flipped; the solution with one variable off by one. Three distinct ones are chosen.
EXP_TEXTS = {
    'integer': [
        'What is {equation}? Round to two decimals.',
        'Compute {equation}, rounded to two decimal places.',
        'Evaluate the expression {equation}. Give the result to two decimals.',
        'A calculation requires the value of {equation}. What is it, rounded to two decimals?',
        'Raise the number to the power shown and round to two decimals: {equation}.',
    ],
    'real_value': [
        'What is the value of {equation}? Round to two decimals.',
        'Compute the power {equation} to two decimal places.',
        'Evaluate {equation}, a power with a fractional exponent. Round to two decimals.',
        'A growth calculation needs the value of {equation}. What is it, to two decimals?',
        'Round the value of {equation} to two decimals.',
    ],
}
SYS_TEXTS = [
    'Solve the following system of equations for {vars}:\n{{equations}}\nWhat is the solution?',
    'Find the values of {vars} that satisfy all of these equations at once:\n{{equations}}',
    'Consider the system\n{{equations}}\nWhich values of {vars} solve it?',
    '{Count} quantities {vars} must satisfy these conditions simultaneously:\n{{equations}}\nWhat are their values?',
    'Which values of {vars} solve the linear system below?\n{{equations}}',
]


def _foundation_templates():
    from . import code_templates as ct
    exp = [ct.CodeTemplate(parts=(t,), type=ty, domain='base', tags='', difficulty_level=0 if ty == 'integer' else 1,
                           spec={'type': ty}) for ty, texts in EXP_TEXTS.items() for t in texts]
    sys_ = []
    for ty, vars_, count in (('two_equation', 'x and y', 'Two'), ('three_equation', 'x, y and z', 'Three')):
        for t in SYS_TEXTS:
            sys_.append(ct.CodeTemplate(parts=(t.format(vars=vars_, Count=count),), type=ty, domain='base', tags='',
                                        difficulty_level=1 if ty == 'two_equation' else 2, spec={'type': ty}))
    ct.register('exponents', exp)
    ct.register('linear_system_eq', sys_)


def _fmt2(x: float) -> str:
    from .utilities.story_utils import num
    return num(x)


@function_registry.register('exponents')
def gen_exponents(metadata):
    from .utilities import filler_utils
    from .utilities.story_utils import pick_apart
    r = filler_utils.rng
    for _ in range(100000):
        base = round(float(r.uniform(1.1, 9.99)), 2)
        if metadata['type'] == 'integer':
            e = int(r.integers(2, 7))
            shown = str(e)
        else:
            e = round(float(r.uniform(0.25, 3.5)), 2)
            if abs(e - round(e)) < 1e-9:
                continue
            shown = _fmt2(e)
        key = base ** e
        if key > 100000:
            continue
        cands = [base * e, base ** (e - 1), base ** (e + 1), base ** (1 / e), e ** base]
        dist = pick_apart(r, key, cands, rel=0.05, lo=0.01, hi=1e6, shown=_fmt2)
        if dist is None:
            continue
        return {'equation': f'{_fmt2(base)}^{shown}', 'options_list': [[_fmt2(x) for x in [key] + dist]],
                'correct_answer': [[1, 0, 0, 0]]}
    raise RuntimeError('exponents: no draw')


def _equation(coefs, const, names) -> str:
    terms = []
    for c, n in zip(coefs, names):
        c = int(c)
        if c == 0:
            continue
        mag = '' if abs(c) == 1 else str(abs(c))
        if not terms:
            terms.append(f"{'-' if c < 0 else ''}{mag}{n}")
        else:
            terms.append(f"{'-' if c < 0 else '+'} {mag}{n}")
    return f"{' '.join(terms)} = {int(const)}"


def _assign(v, names) -> str:
    return ', '.join(f'{n} = {int(x)}' for n, x in zip(names, v))


@function_registry.register('linear_system_eq')
def gen_linear_system_eq(metadata):
    from .utilities import filler_utils
    r = filler_utils.rng
    n = 2 if metadata['type'] == 'two_equation' else 3
    names = ['x', 'y', 'z'][:n]
    for _ in range(100000):
        M = r.integers(-9, 10, size=(n, n))
        if (M == 0).sum() > (1 if n == 3 else 0):
            continue
        det = round(float(np.linalg.det(M)))
        if abs(det) < 2:
            continue
        sol = r.integers(-12, 13, size=n)
        if len(set(int(v) for v in sol)) < n:
            continue  # equal values make the swapped option equal to the key
        b = M @ sol
        if n == 2:
            d = np.array([M[0, 1], -M[0, 0]])             # direction along the first equation's line
        else:
            d = np.cross(M[0], M[1])                       # direction along the first two equations' line
        g = np.gcd.reduce(np.abs(d[d != 0])) if (d != 0).any() else 0
        if g == 0:
            continue
        d = d // g
        t = int(r.choice([-2, -1, 1, 2]))
        other = sol + t * d
        cands = [np.roll(sol, 1) if n == 3 else sol[::-1], other, -sol]
        j = int(r.integers(0, n))
        bump = sol.copy()
        bump[j] += int(r.choice([-1, 1]))
        cands.append(bump)
        texts = [_assign(sol, names)]
        for c in [cands[int(i)] for i in r.permutation(len(cands))]:
            if np.abs(c).max() > 40:
                continue
            tx = _assign(c, names)
            if tx not in texts:
                texts.append(tx)
            if len(texts) == 4:
                break
        if len(texts) < 4:
            continue
        eqs = '\n'.join(_equation(M[i], b[i], names) for i in range(n))
        return {'equations': eqs, 'options_list': [texts], 'correct_answer': [[1, 0, 0, 0]]}
    raise RuntimeError('linear_system_eq: no draw')


@function_registry.register('derivatives')
def gen_derivatives(metadata):
    """
    d/dx of a random linear, quadratic or cubic polynomial with 2-decimal
    coefficients. All options have the same form as the answer; distractors
    are common mistakes (dropping the power-rule factor, keeping the power,
    returning the wrong coefficient).
    """
    def c():
        return round(float(rng.uniform(0.1, 10)), 2)

    r = lambda v: round(v, 2)
    if metadata['type'] == 'linear':
        a, b = c(), c()
        equation = f"{a}x + {b}"
        answer = a
        options = [a, b, r(a + b), r(a * 2)]
    elif metadata['type'] == 'quadratic':
        a, b, k = c(), c(), c()
        equation = f"{a}x^2 + {b}x + {k}"
        answer = f"{r(2*a)}x + {b}"
        options = [answer, f"{a}x + {b}", f"{r(2*a)}x^2 + {b}x", f"{r(2*a)}x + {r(b + k)}"]
    elif metadata['type'] == 'cubic':
        a, b, k, d = c(), c(), c(), c()
        equation = f"{a}x^3 + {b}x^2 + {k}x + {d}"
        answer = f"{r(3*a)}x^2 + {r(2*b)}x + {k}"
        options = [answer, f"{a}x^2 + {b}x + {k}", f"{r(3*a)}x^2 + {b}x + {k}", f"{r(3*a)}x^3 + {r(2*b)}x^2 + {k}x"]
    else:
        raise ValueError(f"Unknown derivatives type {metadata['type']!r}")
    answer_mask, options = gen_answer_and_options(answer, num_options=4, options=options)
    return {
        'equation': equation,
        'options_list': options,
        'correct_answer': answer_mask
    }


_foundation_templates()

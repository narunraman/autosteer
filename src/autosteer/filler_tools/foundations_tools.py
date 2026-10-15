"""constrained_optimization (STEER 1.2.c, Constrained Optimization) and a helper for discrete optimisation.

Decision of 5 October 2026: the class ConstrainedOptimizationProblem (from STEER-generation@legacy
template_configs/foundations_utils.py, never registered) is fixed and registered as the filler of the data
directory's constrained_optimization templates. The old class passed the unnegated objective to linprog for a
maximisation (so the "maximum" was the minimum), printed only the diagonal of the quadratic term while solving with
the full matrix, and drew real coefficients in [0, 1] for which the minimum of a positive objective over x >= 0 was
always 0 at the origin. Now, with two variables x, y >= 0 and whole-number coefficients:

  linear (template tag `linear`, difficulty 4): max c.x subject to two constraints a_i.x <= b_i, or min c.x subject to
      two constraints a_i.x >= b_i (c, a_i > 0). The optimum is found by enumerating the vertices of the feasible
      region and is unique (the runner-up vertex at least 0.5 worse); the two constraint lines meet inside the
      feasible quadrant, so neither constraint is redundant; the solution is cross-checked with linprog
      (objective negated for a maximisation). Distractors: the objective at the other vertices (not the origin); the optimum with only
      the first or only the second constraint.
  quadratic (tag `quadratic`, difficulty 5): max -p x^2 - q y^2 + a x + b y subject to u x + v y <= s, or
      min p x^2 + q y^2 - a x - b y subject to u x + v y >= s (p, q > 0), with the unconstrained optimum infeasible,
      so the constraint binds; the optimum follows from the KKT conditions and is interior to the segment.
      Distractors: the unconstrained optimum (constraint ignored); the objective at the two ends of the constraint
      segment; at its midpoint.
The key is the optimal value (2 decimals). Three distractors at least 5% apart from the key and from each other.

Templates (October 2026 cleanup): the data directory's templates are no longer used. Their stories (a garden, a
project budget) were decoration around a bare formula: nothing in the story fixed the objective or the constraints.
The templates are now written here (code templates):
  linear: word problems whose story determines the problem. Type max: a producer (farm, steel mill, refinery,
      juice plant, fertilizer plant) chooses divisible amounts x, y of two products; the profit per unit is the
      objective's coefficients, and two resources (hours of two machines, labor and fertilizer) with a stated use
      per unit and a stated amount available are the two "<=" constraints. Type min: a buyer or operator (feed
      mix, two mines, a supplement mix, two presses, two power plants) chooses divisible amounts of two inputs; the
      cost per unit is the objective, and two requirements (protein and energy, two ore grades, ...) with stated
      contents per unit and a stated minimum are the two ">=" constraints. Money is shown in dollars (some stories
      count hundreds of dollars, e.g. $700 per hour).
  quadratic: the problem stated bare, without a story ("Maximize f(x, y) = ... subject to ..."), three phrasings.
"""
from fractions import Fraction
from itertools import combinations
from string import ascii_uppercase

import numpy as np
from scipy.optimize import linprog

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import num, pick_apart
from autosteer.filler_tools.rng import pyrng

ALPHABET = list(ascii_uppercase)
function_registry = FillRegister()


def _lin(coefs, names=('x', 'y')) -> str:
    out = []
    for c, n in zip(coefs, names):
        c = int(c)
        if c == 0:
            continue
        mag = '' if abs(c) == 1 else str(abs(c))
        out.append((f"{'-' if c < 0 else ''}{mag}{n}") if not out else f"{'-' if c < 0 else '+'} {mag}{n}")
    return ' '.join(out)


class ConstrainedOptimizationProblem:
    """A two-variable problem with whole-number data (x, y >= 0), linear (LP) or quadratic (QP) objective."""

    def __init__(self, rng=None, maximize=False, problem_type='LP'):
        self.rng = rng if rng is not None else filler_utils.rng
        self.maximize = maximize
        self.problem_type = problem_type.upper()
        self.generate_problem()

    # ------------------------------------------------------------------ problem generation
    def generate_problem(self):
        r = self.rng
        for _ in range(100000):
            if self.problem_type == 'LP':
                self.c = [int(v) for v in r.integers(1, 10, size=2)]
                self.A = [[int(v) for v in r.integers(1, 10, size=2)] for _ in range(2)]
                self.b = [int(v) for v in r.integers(10, 61, size=2)]
                if self.A[0][0] * self.A[1][1] == self.A[0][1] * self.A[1][0]:
                    continue
                verts = self.vertices()
                (a1, a2), (a3, a4) = self.A
                det = a1 * a4 - a2 * a3
                cross = (Fraction(self.b[0] * a4 - self.b[1] * a2, det), Fraction(a1 * self.b[1] - a3 * self.b[0], det))
                if cross not in verts:
                    continue                                          # both constraints must shape the region
                vals = sorted((self.value(v), v) for v in verts)
                if len(vals) < 3:
                    continue
                best, second = (vals[-1], vals[-2]) if self.maximize else (vals[0], vals[1])
                if abs(best[0] - second[0]) < 0.5:
                    continue
                self.x_opt, self.v_opt = best[1], best[0]
                return
            p, q = int(r.integers(1, 4)), int(r.integers(1, 4))
            a, b = int(r.integers(4, 31)), int(r.integers(4, 31))
            u, v = int(r.integers(1, 6)), int(r.integers(1, 6))
            xu, yu = Fraction(a, 2 * p), Fraction(b, 2 * q)          # unconstrained optimum
            if self.maximize:
                s = int(r.integers(2, 41))
                if u * xu + v * yu <= s:
                    continue                                          # constraint must bind
                lam = (u * xu + v * yu - s) / (Fraction(u * u, 2 * p) + Fraction(v * v, 2 * q))
                x, y = xu - lam * Fraction(u, 2 * p), yu - lam * Fraction(v, 2 * q)
            else:
                s = int(r.integers(10, 81))
                if u * xu + v * yu >= s:
                    continue
                lam = (s - u * xu - v * yu) / (Fraction(u * u, 2 * p) + Fraction(v * v, 2 * q))
                x, y = xu + lam * Fraction(u, 2 * p), yu + lam * Fraction(v, 2 * q)
            if x <= 0 or y <= 0:
                continue
            self.p, self.q, self.a, self.b, self.u, self.v, self.s = p, q, a, b, u, v, s
            self.x_unc = (xu, yu)
            self.x_opt = (x, y)
            self.v_opt = self.value(self.x_opt)
            return
        raise RuntimeError('constrained_optimization: no problem drawn')

    # ------------------------------------------------------------------ evaluation
    def value(self, pt):
        x, y = (Fraction(t) for t in pt)
        if self.problem_type == 'LP':
            return self.c[0] * x + self.c[1] * y
        sign = -1 if self.maximize else 1
        return sign * (self.p * x * x + self.q * y * y) + (-1 if not self.maximize else 1) * (self.a * x + self.b * y)

    def feasible(self, pt) -> bool:
        x, y = pt
        if x < 0 or y < 0:
            return False
        for (a1, a2), bi in zip(self.A, self.b):
            lhs = a1 * x + a2 * y
            if (self.maximize and lhs > bi) or (not self.maximize and lhs < bi):
                return False
        return True

    def vertices(self):
        """Vertices of the feasible region: pairwise intersections of the constraint lines and the axes."""
        lines = [((a1, a2), bi) for (a1, a2), bi in zip(self.A, self.b)] + [((1, 0), 0), ((0, 1), 0)]
        out = []
        for (n1, b1), (n2, b2) in combinations(lines, 2):
            det = n1[0] * n2[1] - n1[1] * n2[0]
            if det == 0:
                continue
            x = Fraction(b1 * n2[1] - b2 * n1[1], det)
            y = Fraction(n1[0] * b2 - n2[0] * b1, det)
            if self.feasible((x, y)) and (x, y) not in out:
                out.append((x, y))
        return out

    def find_optimal(self):
        """(x, value) from scipy: linprog for LP (objective negated for a maximisation), as a cross-check."""
        if self.problem_type != 'LP':
            return tuple(float(t) for t in self.x_opt), float(self.v_opt)
        sign = -1 if self.maximize else 1
        A = np.array(self.A, dtype=float) * (1 if self.maximize else -1)
        b = np.array(self.b, dtype=float) * (1 if self.maximize else -1)
        res = linprog(c=sign * np.array(self.c, dtype=float), A_ub=A, b_ub=b, bounds=[(0, None)] * 2,
                      method='highs')
        if not res.success:
            raise ValueError('Optimization failed')
        return res.x, sign * res.fun

    # ------------------------------------------------------------------ text and distractors
    def objective_text(self) -> str:
        if self.problem_type == 'LP':
            return _lin(self.c)
        if self.maximize:
            quad = f"-{'' if self.p == 1 else self.p}x^2 - {'' if self.q == 1 else self.q}y^2"
            return f'{quad} + {self.a}x + {self.b}y'
        quad = f"{'' if self.p == 1 else self.p}x^2 + {'' if self.q == 1 else self.q}y^2"
        return f'{quad} - {self.a}x - {self.b}y'

    def constraints_text(self) -> str:
        op = '<=' if self.maximize else '>='
        if self.problem_type == 'LP':
            cons = [f'{_lin(a)} {op} {bi}' for a, bi in zip(self.A, self.b)]
        else:
            cons = [f'{_lin((self.u, self.v))} {op} {self.s}']
        return ', '.join(cons + ['x >= 0']) + ' and y >= 0'

    def distractor_values(self):
        if self.problem_type == 'LP':
            others = [self.value(v) for v in self.vertices() if v != self.x_opt and self.value(v) != 0]
            single = []
            for i in range(2):
                sub = ConstrainedOptimizationProblem.__new__(ConstrainedOptimizationProblem)
                sub.problem_type, sub.maximize, sub.c = 'LP', self.maximize, self.c
                sub.A, sub.b = [self.A[i]], [self.b[i]]
                vals = [sub.value(v) for v in sub.vertices()]
                if vals:
                    single.append(max(vals) if self.maximize else min(vals))
            return [float(v) for v in others + single]
        ends = [(Fraction(self.s, self.u), 0), (0, Fraction(self.s, self.v))]
        mid = ((ends[0][0] + ends[1][0]) / 2, (ends[0][1] + ends[1][1]) / 2)
        return [float(self.value(pt)) for pt in [self.x_unc] + ends + [mid]]

    def __str__(self):
        kind = 'Maximize' if self.maximize else 'Minimize'
        return f'{kind} {self.objective_text()} subject to {self.constraints_text()}'


# Story LPs. Each story: domain, who (anon intro / second-person intro / subject words), the two variables and the
# two constraints, all in fixed sentences; the numbers are slots. scale: dollars per shown coefficient unit.
LP_MAX_STORIES = [
    dict(domain='agriculture', anon='A farm grows wheat and barley.', you='You run a farm that grows wheat and barley.',
         x='wheat', y='barley', unit='acre', verb='plant', r1=('hours', 'field labor'), r2=('bags', 'fertilizer'),
         period='this season', scale=100),
    dict(domain='manufacturing', anon='A steel mill makes sheet steel and steel bars.',
         you='You run a steel mill that makes sheet steel and steel bars.', x='sheet steel', y='steel bars',
         unit='tonne', verb='make', r1=('hours', 'furnace time'), r2=('hours', 'rolling-mill time'),
         period='this week', scale=100),
    dict(domain='energy', anon='A small refinery produces gasoline and diesel.',
         you='You run a small refinery that produces gasoline and diesel.', x='gasoline', y='diesel',
         unit='thousand barrels', verb='produce', r1=('hours', 'distillation-unit time'), r2=('hours', 'cracker time'),
         period='this week', scale=1000),
    dict(domain='food', anon='A juice plant presses apple juice and orange juice.',
         you='You run a juice plant that presses apple juice and orange juice.', x='apple juice', y='orange juice',
         unit='thousand litres', verb='produce', r1=('hours', 'pressing time'), r2=('hours', 'bottling-line time'),
         period='this week', scale=100),
    dict(domain='chemicals', anon='A fertilizer plant makes nitrate fertilizer and phosphate fertilizer.',
         you='You run a fertilizer plant that makes nitrate fertilizer and phosphate fertilizer.',
         x='nitrate fertilizer', y='phosphate fertilizer', unit='tonne', verb='make', r1=('hours', 'reactor time'),
         r2=('hours', 'dryer time'), period='this week', scale=100),
]
LP_MIN_STORIES = [
    dict(domain='agriculture', anon='A cattle farm mixes its feed from corn and soybean meal.',
         you='You mix the feed for a cattle farm from corn and soybean meal.', x='corn', y='soybean meal',
         unit='sack', per='Each sack of {v} costs ${c} and provides {a1} units of protein and {a2} units of energy',
         need='The day\'s feed must provide at least {b1} units of protein and at least {b2} units of energy',
         amounts='any amount of each, including part of a sack', scale=1, what='feed'),
    dict(domain='mining', anon='A mining company runs two mines, North and South.',
         you='You run a mining company with two mines, North and South.', x='North', y='South',
         unit='hour', per='Each hour of operating the {v} mine costs ${c} and yields {a1} tonnes of high-grade ore and '
         '{a2} tonnes of low-grade ore', need='A contract requires at least {b1} tonnes of high-grade ore and at least '
         '{b2} tonnes of low-grade ore this week', amounts='any number of hours, including fractions of an hour',
         scale=100, what='operating the mines'),
    dict(domain='nutrition', anon='A hospital dietitian prepares a daily drink from two liquid supplements, A and B.',
         you='You prepare a patient\'s daily drink from two liquid supplements, A and B.', x='supplement A',
         y='supplement B', unit='100 ml', per='Each 100 ml of {v} costs ${c} and provides {a1} grams of protein and '
         '{a2} milligrams of iron', need='The drink must provide at least {b1} grams of protein and at least {b2} '
         'milligrams of iron', amounts='any amount of each', scale=1, what='the drink'),
    dict(domain='manufacturing', anon='A print shop runs two presses, a small one and a large one.',
         you='You run a print shop with two presses, a small one and a large one.', x='small press', y='large press',
         unit='hour', per='Each hour of running the {v} costs ${c} and prints {a1} thousand flyers and {a2} thousand '
         'posters', need='An order calls for at least {b1} thousand flyers and at least {b2} thousand posters',
         amounts='any number of hours, including fractions of an hour', scale=100, what='running the presses'),
    dict(domain='energy', anon='A utility runs two plants that both produce electricity and steam.',
         you='You run a utility with two plants that both produce electricity and steam.', x='first plant',
         y='second plant', unit='hour', per='Each hour of running the {v} costs ${c} and produces {a1} megawatt-hours '
         'of electricity and {a2} tonnes of steam', need='Tomorrow the utility must supply at least {b1} '
         'megawatt-hours of electricity and at least {b2} tonnes of steam', amounts='any number of hours, including '
         'fractions of an hour', scale=1000, what='running the plants'),
]
QP_STEMS = {
    'max': ['Maximize f(x, y) = {objective_function} subject to {constraints}. What is the maximum value of f?',
            'Consider the problem of maximizing {objective_function} over x and y, subject to {constraints}. What is '
            'the maximum value of the objective?',
            'Solve the following optimization problem. Maximize {objective_function} subject to the constraints '
            '{constraints}. What is the maximum value?'],
    'min': ['Minimize f(x, y) = {objective_function} subject to {constraints}. What is the minimum value of f?',
            'Consider the problem of minimizing {objective_function} over x and y, subject to {constraints}. What is '
            'the minimum value of the objective?',
            'Solve the following optimization problem. Minimize {objective_function} subject to the constraints '
            '{constraints}. What is the minimum value?'],
}


def _cap(t: str) -> str:
    return t[0].upper() + t[1:]


def _lp_max_text(st: dict, you: bool, stem: int) -> str:
    u, (n1, r1), (n2, r2) = st['unit'], st['r1'], st['r2']
    intro = st['you'] if you else st['anon']
    who = 'You' if you else 'It'
    text = (f"{intro} {who} can {st['verb']} any amounts of {st['x']} and {st['y']}, including fractions of a {u}. "
            f"Each {u} of {st['x']} brings a profit of ${{c1}} and uses {{a11}} {n1} of {r1} and {{a21}} {n2} of {r2}; "
            f"each {u} of {st['y']} brings a profit of ${{c2}} and uses {{a12}} {n1} of {r1} and {{a22}} {n2} of {r2}. "
            f"{_cap(st['period'])} {{b1}} {n1} of {r1} and {{b2}} {n2} of {r2} are available.")
    if stem == 0:
        q = f" What is the maximum profit {'you' if you else 'it'} can make {st['period']}?"
    else:
        q = (f" If {'you choose' if you else 'it chooses'} the amounts of {st['x']} and {st['y']} to maximize "
             f"profit, what is the maximum profit {st['period']}?")
    return text + q


def _lp_min_text(st: dict, you: bool, stem: int) -> str:
    intro = st['you'] if you else st['anon']
    who = 'You' if you else 'It'
    s1 = st['per'].format(v=st['x'], c='{c1}', a1='{a11}', a2='{a21}')
    s2 = st['per'].format(v=st['y'], c='{c2}', a1='{a12}', a2='{a22}')
    need = st['need'].format(b1='{b1}', b2='{b2}')
    text = f"{intro} {who} can use {st['amounts']}. {s1}; {s2[0].lower() + s2[1:]}. {need}."
    if stem == 0:
        q = f" What is the minimum cost of {st['what']} that meets these requirements?"
    else:
        q = f" Meeting these requirements at the lowest possible cost, what is the minimum cost of {st['what']}?"
    return text + q


def _co_templates() -> list:
    out = []
    for qtype, stories, render in (('max', LP_MAX_STORIES, _lp_max_text), ('min', LP_MIN_STORIES, _lp_min_text)):
        for k, st in enumerate(stories):
            for you in (False, True):
                for stem in (0, 1):
                    out.append(ct.CodeTemplate(parts=(render(st, you, stem),), type=qtype, domain=st['domain'],
                                               tags='linear;' + ('second_person' if you else 'anon'),
                                               difficulty_level=4, spec={'kind': 'story', 'story': k}))
        for stem in QP_STEMS[qtype]:
            out.append(ct.CodeTemplate(parts=(stem,), type=qtype, domain='base', tags='quadratic',
                                       difficulty_level=5, spec={'kind': 'bare'}))
    return out


def _money(v: float) -> str:
    return '$' + num(v)


@function_registry.register('constrained_optimization')
def fill_constrained_optimization(metadata):
    r = filler_utils.rng
    spec = ct.spec_for('constrained_optimization', metadata['question_text'])
    maximize = metadata['type'] == 'max'
    if spec['kind'] == 'story':
        st = (LP_MAX_STORIES if maximize else LP_MIN_STORIES)[spec['story']]
        scale = st['scale']
        for _ in range(1000):
            prob = ConstrainedOptimizationProblem(r, maximize=maximize, problem_type='LP')
            if min(min(row) for row in prob.A) < 2:
                continue  # the story would say "1 hours"
            key = float(prob.v_opt) * scale
            dist = pick_apart(r, key, [v * scale for v in prob.distractor_values()], rel=0.05, lo=0,
                              shown=_money)
            if dist is None:
                continue
            (a11, a12), (a21, a22) = prob.A
            out = dict(c1=str(prob.c[0] * scale), c2=str(prob.c[1] * scale), a11=str(a11), a12=str(a12),
                       a21=str(a21), a22=str(a22), b1=str(prob.b[0]), b2=str(prob.b[1]))
            out.update(options_list=[[_money(v) for v in [key] + dist]], correct_answer=[[1, 0, 0, 0]])
            return out
        raise RuntimeError('constrained_optimization: no question drawn')
    for _ in range(1000):
        prob = ConstrainedOptimizationProblem(r, maximize=maximize, problem_type='QP')
        key = float(prob.v_opt)
        dist = pick_apart(r, key, prob.distractor_values(), rel=0.05, lo=-1e6, shown=num)
        if dist is None:
            continue
        return {'objective_function': prob.objective_text(), 'constraints': prob.constraints_text(),
                'options_list': [[num(v) for v in [key] + dist]], 'correct_answer': [[1, 0, 0, 0]]}
    raise RuntimeError('constrained_optimization: no question drawn')


ct.register('constrained_optimization', _co_templates())


def discrete_opt_values(metadata):
    num_options = 2 if 'two_options' in metadata['tags'] else 3

    if metadata['domain'] == 'base':
        random_numbers = pyrng.sample(range(-100, 100), num_options)
    elif metadata['domain'] == 'sports':
        random_numbers = pyrng.sample(range(1, 100), num_options)
    elif metadata['domain'] == 'finance':
        random_numbers = [round(pyrng.uniform(0, 1), 2) for _ in range(num_options)]
    else:
        raise ValueError(f"Unknown domain {metadata['domain']}")

    if metadata['type'] == 'min':
        random_numbers = sorted(random_numbers, reverse=True)
    elif metadata['type'] == 'max':
        random_numbers = sorted(random_numbers)
    else:
        raise ValueError(f"Unknown type {metadata['type']}")
    
            
    return {f"number_{i+1}": number for i, number in enumerate(random_numbers)}
"""mrs_tangency (STEER-ME, Consumption Decisions, Properties of Utility Functions): Tangency and the Marginal Rate
of Substitution ("the MRS between two goods at a given point in the budget constraint as tangent to the
indifference curve").

Decision of 5 October 2026; templates written here (the old `mrs_tangency` stub in micro_utility_tools.py had no
body). A consumer with Cobb-Douglas utility U(x, y) = x^a y^b spends income m on two divisible goods with prices px,
py (the consumers of hicksian_duality). The MRS is MU_x / MU_y = a y / (b x), the amount of y the consumer would give
up for one more unit of x; at the utility-maximising bundle it equals the price ratio px / py (tangency).

Types:
  mrs_at_optimum: the MRS at the chosen (utility-maximising) bundle. Key px / py (2 decimals). Distractors: the
      inverse ratio py / px; the MRS at the bundle that splits spending equally (a px / (b py)); the ratio of the
      exponents a / b (quantities left out); the MRS at the consumer's current bundle (stated in the text).
  tangency_bundle: four bundles that all cost exactly m; the key is the tangency point (the Marshallian bundle).
      Distractors: equal spending on both goods; the exponents swapped; the tangency condition with the price ratio
      inverted (a y / (b x) = py / px).
  adjust: a stated bundle on the budget line; options "spend more on x and less on y", "spend less on x and more on y",
      "keep this bundle". Key: more x if MRS > px / py, less x if MRS < px / py, keep if the bundle is the tangency point (a
      third of the questions; the parameters are then chosen so that the optimum has at most 2 decimals). Where
      possible, comparing the MRS with the inverted price ratio would give another answer.
Distractor numbers and bundles are at least 10% apart from the key and from each other (bundles: in one good).

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .hicksian_duality_tools import STORIES, marshallian
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart, voice

function_registry = FillRegister()

ELEMENT = 'mrs_tangency'
TYPES = ('mrs_at_optimum', 'tangency_bundle', 'adjust')
MORE, LESS, KEEP = ('Spend more on {gx} and less on {gy}', 'Spend less on {gx} and more on {gy}',
                    'Keep this bundle: no affordable bundle gives higher utility')


def _voice(tag, k, st):
    v = voice(tag, k, f"the {st['anon'].split(' ', 1)[1]}")
    v['first'] = st['anon'][0].upper() + st['anon'][1:] if tag == 'anon' else v['Subj']
    if tag == 'anon':
        v['poss'], v['Poss'] = 'their', 'Their'
    return v


def _parts(st: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    v = _voice(tag, k, st)
    text = (f"{v['first']} spend{v['s']} ${{income}} a {st['per']} on {st['gx']} (good x) and {st['gy']} (good y). "
            f"{v['Poss']} utility is U(x, y) = {{ufunc}}, where x is the number of {st['uxs']} and y the number of "
            f"{st['uys']} per {st['per']}. The price of {st['gx']} is ${{px}} per {st['ux']} and the price of "
            f"{st['gy']} is ${{py}} per {st['uy']}; both can be bought in any fractional amount.")
    if qtype == 'mrs_at_optimum':
        text += (f" {v['Subj']} currently buy{v['s']} {{xnow}} {st['uxs']} and {{ynow}} {st['uys']}, but {v['subj']} "
                 f"will switch to the affordable bundle that maximizes {v['poss']} utility. At that bundle, what is "
                 f"{v['poss']} marginal rate of substitution MRS = MU_x / MU_y, the amount of good y {v['subj']} would "
                 f"give up for one more unit of good x? Round to two decimals.")
    elif qtype == 'tangency_bundle':
        text += (f" Each of the following bundles spends the whole ${{income}} (amounts rounded to two decimals). At "
                 f"which of them is {v['poss']} indifference curve tangent to the budget line?")
    else:
        text += (f" {v['Subj']} currently buy{v['s']} {{xnow}} {st['uxs']} and {{ynow}} {st['uys']}, which costs "
                 f"exactly ${{income}}. Comparing {v['poss']} marginal rate of substitution at this bundle with the "
                 f"price ratio, what should {v['subj']} do to raise {v['poss']} utility without spending more?")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(st, tag, k, qtype), type=qtype, domain=st['domain'],
                                           tags=tag, difficulty_level=1, spec={'story': k, 'type': qtype}))
    return out


# ------------------------------------------------------------------------------------------------------------------
def mrs(a, b, x, y):
    return a * y / (b * x)


def _exact(v: float) -> bool:
    return abs(v * 100 - round(v * 100)) < 1e-7


def bundle_text(x, y) -> str:
    return f'x = {num(x)}, y = {num(y)}'


def _bundles_apart(p, q):
    return any(abs(p[i] - q[i]) >= 0.1 * max(abs(p[i]), abs(q[i])) for i in (0, 1))


def _params(r, exact: bool):
    while True:
        a = round(0.05 * int(r.integers(4, 17)), 2)
        b = round(0.05 * int(r.integers(4, 17)), 2)
        if a == b or not 0.6 <= a + b <= 1.4:
            continue
        px, py = int(r.integers(2, 31)), int(r.integers(2, 31))
        if px == py:
            continue
        m = 10 * int(r.integers(10, 81))
        x0, y0 = marshallian(a, b, px, py, m)
        if not (2 <= x0 <= 300 and 2 <= y0 <= 300):
            continue
        if exact and not (_exact(x0) and _exact(y0)):
            continue
        return a, b, px, py, m, x0, y0


def _off_bundle(r, px, py, m, x0):
    """An integer amount of x away from the optimum, with y on the budget line at 2 exact decimals."""
    for _ in range(200):
        f = float(r.uniform(0.45, 0.8)) if r.integers(0, 2) == 0 else float(r.uniform(1.25, 1.7))
        x = int(round(x0 * f))
        y = (m - px * x) / py
        if x >= 1 and y >= 1 and _exact(y) and abs(x - x0) >= 0.1 * x0:
            return x, y
    return None


def draw(r, qtype: str) -> dict:
    for _ in range(100000):
        keep = qtype == 'adjust' and r.random() < 1 / 3
        a, b, px, py, m, x0, y0 = _params(r, exact=keep)
        out = {'ufunc': f'x^{a:g} * y^{b:g}', 'px': str(px), 'py': str(py), 'income': f'{m:,}'}
        if qtype == 'mrs_at_optimum':
            now = _off_bundle(r, px, py, m, x0)
            if now is None:
                continue
            key = px / py
            cands = [py / px, a * px / (b * py), a / b, mrs(a, b, *now)]
            dist = pick_apart(r, key, cands, lo=0.005, shown=num)
            if dist is None:
                continue
            opts = [num(z) for z in [key] + dist]
            out.update(xnow=num(now[0]), ynow=num(now[1]))
        elif qtype == 'tangency_bundle':
            xi = m / (px + b * py ** 2 / (a * px))     # a y / (b x) = py / px on the budget line
            cands = [(m / (2 * px), m / (2 * py)), marshallian(b, a, px, py, m), (xi, (m - px * xi) / py)]
            allb = [(x0, y0)] + cands
            if not all(_bundles_apart(allb[i], allb[j]) for i in range(4) for j in range(i)):
                continue
            opts = [bundle_text(*z) for z in [(x0, y0)] + cands]
        else:
            if keep:
                now, ans = (x0, y0), KEEP
            else:
                now = _off_bundle(r, px, py, m, x0)
                if now is None:
                    continue
                g = mrs(a, b, *now)
                ans = MORE if g > px / py else LESS
                wrong = MORE if g > py / px else LESS
                if wrong == ans and r.random() < 0.8:
                    continue  # prefer bundles where comparing with the inverted price ratio misleads
            opts = [ans] + [o for o in (MORE, LESS, KEEP) if o != ans]
            out.update(xnow=num(now[0]), ynow=num(now[1]))
        if len(set(opts)) != len(opts):
            continue
        out.update(options_list=[opts], correct_answer=[[1] + [0] * (len(opts) - 1)])
        return out
    raise RuntimeError(f'mrs_tangency: no draw for {qtype}')


@function_registry.register(ELEMENT)
def fill_mrs_tangency(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    st = STORIES[spec['story']]
    out = draw(filler_utils.rng, spec['type'])
    if spec['type'] == 'adjust':
        out['options_list'] = [[o.format(gx=st['gx'], gy=st['gy']) for o in out['options_list'][0]]]
    return out


ct.register(ELEMENT, _templates())

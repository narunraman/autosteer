"""consumption_changes (STEER-ME, Consumption Decisions, Comparative Statics of Demand): Consumption Changes (the paper's definition, paraphrased: how
consumption of a good changes when its price changes).

Decision of 7 October 2026; templates written here (the old notebook's three hand-written questions had economically
wrong keys: spending on a good whose price rose does not have to fall). The Giffen half of the paper's definition is
left out: it has no determinate key without a specific Giffen utility (triage D8).

A consumer (the consumers of hicksian_duality) spends all income on two goods with a CES utility
U = (a x^ρ + b y^ρ)^(1/ρ), a + b = 1, whose elasticity of substitution σ = 1/(1 - ρ) is stated, or a Cobb-Douglas
utility x^a y^b (σ = 1). The expenditure share of x is

    s_x = 1 / (1 + (b/a)^σ (py/px)^(1 - σ)),

so when px rises s_x rises if σ < 1, stays the same if σ = 1 and falls if σ > 1.

Types:
  direction: the price of x rises by k%; does the share of income spent on x rise, fall or stay the same? (a third
      each of σ < 1, σ = 1, σ > 1).
  new_share: the same change; the share of income spent on x afterwards, in percent (2 decimals). Distractors: the
      old share, the quantity of x left unchanged (old share times 1 + k%), the old share divided by 1 + k%, the share
      of y, the share if the price had fallen by k%. At least 10% apart from the key and from each other.

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .hicksian_duality_tools import STORIES
from .utilities import filler_utils
from .utilities.story_utils import PERSPECTIVES, num, pick_apart, voice

function_registry = FillRegister()

ELEMENT = 'consumption_changes'
TYPES = ('direction', 'new_share')
RISES, FALLS, SAME = 'It rises', 'It falls', 'It stays the same'
RHOS = (-4, -3, -1, -0.25, 0.2, 0.5, 0.6, 0.75)        # sigma 0.2, 0.25, 0.5, 0.8, 1.25, 2, 2.5, 4


def _voice(tag, k, st):
    v = voice(tag, k, f"the {st['anon'].split(' ', 1)[1]}")
    v['first'] = st['anon'][0].upper() + st['anon'][1:] if tag == 'anon' else v['Subj']
    if tag == 'anon':
        v['poss'], v['Poss'] = 'their', 'Their'
    return v


def _parts(st: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    v = _voice(tag, k, st)
    text = (f"{v['first']} spend{v['s']} all of {v['poss']} income of ${{income}} a {st['per']} on {st['gx']} (good x) "
            f"and {st['gy']} (good y). {v['Poss']} utility is {{ufunc}}, where x is the number of {st['uxs']} and y "
            f"the number of {st['uys']} per {st['per']}; this is {{family}} utility function, whose elasticity of "
            f"substitution between the two goods is σ = {{sigma}}. The price of {st['gx']} is ${{px}} per {st['ux']} "
            f"and the price of {st['gy']} is ${{py}} per {st['uy']}, and both can be bought in any fractional amount. "
            f"The price of {st['gx']} now rises by {{k}}%, while the price of {st['gy']} and {v['poss']} income stay "
            f"the same, and {v['subj']} again choose{v['s']} the bundle that maximizes {v['poss']} utility.")
    if qtype == 'direction':
        text += f" What happens to the share of {v['poss']} income that {v['subj']} spend{v['s']} on {st['gx']}?"
    else:
        text += (f" What percentage of {v['poss']} income will {v['subj']} spend on {st['gx']} after the price rise? "
                 f"Round to two decimals.")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(st, tag, k, qtype), type=qtype, domain=st['domain'],
                                           tags=tag, difficulty_level=1 if qtype == 'direction' else 2,
                                           spec={'story': k, 'type': qtype}))
    return out


def share_x(a: float, sigma: float, px: float, py: float) -> float:
    """Expenditure share of x with weights a, 1 - a and elasticity of substitution sigma."""
    b = 1 - a
    return 1 / (1 + (b / a) ** sigma * (py / px) ** (1 - sigma))


def draw(r, qtype: str) -> dict:
    for _ in range(100000):
        case = int(r.integers(0, 3))                    # 0: sigma < 1, 1: Cobb-Douglas, 2: sigma > 1
        a = round(0.05 * int(r.integers(4, 17)), 2)     # 0.2 .. 0.8
        b = round(1 - a, 2)
        if case == 1:
            sigma = 1.0
            ufunc, family = f'U(x, y) = x^{a:g} * y^{b:g}', 'a Cobb-Douglas'
        else:
            rho = float(r.choice([z for z in RHOS if (z < 0) == (case == 0)]))
            sigma = 1 / (1 - rho)
            ufunc = (f'U(x, y) = (a * x^ρ + b * y^ρ)^(1/ρ) with a = {a:g}, b = {b:g} and ρ = {rho:g}')
            family = 'a CES (constant elasticity of substitution)'
        px, py = int(r.integers(2, 41)), int(r.integers(2, 41))
        k = 5 * int(r.integers(2, 13))                  # 10% .. 60%
        m = 10 * int(r.integers(20, 301))
        s0 = share_x(a, sigma, px, py)
        s1 = share_x(a, sigma, px * (1 + k / 100), py)
        out = {'ufunc': ufunc, 'family': family, 'sigma': num(sigma), 'px': str(px), 'py': str(py), 'k': str(k),
               'income': f'{m:,}'}
        if qtype == 'direction':
            if case != 1 and abs(s1 - s0) < 0.005:
                continue
            key = (RISES, SAME, FALLS)[case]
            opts = [key] + [o for o in (RISES, FALLS, SAME) if o != key]
        else:
            key = 100 * s1
            cands = [100 * s0, 100 * s0 * (1 + k / 100), 100 * s0 / (1 + k / 100), 100 - key,
                     100 * share_x(a, sigma, px * (1 - k / 100), py)]
            dist = pick_apart(r, key, cands, lo=0.5, hi=99.5, shown=num)
            if dist is None or not 1 <= key <= 99:
                continue
            opts = [f'{num(z)}%' for z in [key] + dist]
        out.update(options_list=[opts], correct_answer=[[1] + [0] * (len(opts) - 1)])
        return out
    raise RuntimeError(f'consumption_changes: no draw for {qtype}')


@function_registry.register(ELEMENT)
def fill_consumption_changes(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return draw(filler_utils.rng, spec['type'])


ct.register(ELEMENT, _templates())

"""income_elasticity (STEER-ME, Consumption Decisions, Comparative Statics of Demand): Income Elasticity of Demand.

Decision of 5 October 2026 (triage D10): computed from arithmetic on percentage changes, as the released
price_elasticity; templates written here. The old filler (`gen_income_elasticity_values` in
comparative_statics_demand_tools.py, no templates) is replaced; its `direct` type is kept, and its `quantity` and
`income` types, which multiplied an elasticity by a change that was not a percentage (and, for `income`, multiplied
where it should divide), now state and ask for percentage changes. (With Cobb-Douglas utility the income elasticity
is always 1, so a utility-based design would not discriminate.)

Types (the consumers and goods of hicksian_duality; prices unchanged throughout):
  direct: income from m1 to m2 and the quantity bought from q1 to q2; key E = ((q2 - q1) / q1) / ((m2 - m1) / m1),
      percentage changes from the original values (2 decimals). A quarter of the goods are inferior (q falls).
      Distractors: the ratio inverted; the midpoint (arc) formula; the slope (q2 - q1) / (m2 - m1); the changes
      measured from the new values.
  quantity: the income elasticity E and a k% rise in income; by what percentage does the quantity change? Key E k.
      Distractors: k / E, E + k, E k / 100.
  income: the income elasticity E and a k% change in the quantity; by what percentage did income change? Key k / E.
      Distractors: E k, k - E, E / k.
In half the draws no option is a sign flip of another; in the other half the options are the key, -key, one mistake
and its sign flip (sign_balanced; review round 3: -key was the only +/- pair).
Options are numbers (direct) or signed percentages "+4.5%" / "-2.25%" (2 decimals), at least 10% apart from the key and
from each other.

Perspectives (tags): first_person, second_person, man, woman, anon.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .hicksian_duality_tools import STORIES
from .utilities import filler_utils
from .utilities.rounding import near_tie
from .utilities.story_utils import PERSPECTIVES, num, pick_apart, voice

function_registry = FillRegister()

ELEMENT = 'income_elasticity'
TYPES = ('direct', 'quantity', 'income')


def _parts(st: dict, tag: str, k: int, qtype: str) -> tuple[str]:
    v = voice(tag, k, f"the {st['anon'].split(' ', 1)[1]}")
    first = st['anon'][0].upper() + st['anon'][1:] if tag == 'anon' else v['Subj']
    poss = 'their' if tag == 'anon' else v['poss']
    per = st['per']
    if qtype == 'direct':
        text = (f"When {poss} income changed from ${{income_start}} to ${{income_end}} a {per}, with prices unchanged, "
                f"{v['subj'] if tag != 'anon' else st['anon']} changed {poss} purchases of {st['gx']} from "
                f"{{quantity_start}} to {{quantity_end}} {st['uxs']} a {per}. Measuring each percentage change from "
                f"the original value, what is {poss} income elasticity of demand for {st['gx']}? Round to two "
                f"decimals.")
        return (text[0].upper() + text[1:],)
    text = (f"{first} buy{v['s']} {st['gx']} every {per}, and {poss} income elasticity of demand for {st['gx']} is "
            f"{{elasticity}}. Prices do not change. Use the approximation percentage change in quantity demanded = income "
            f"elasticity x percentage change in income.")
    if qtype == 'quantity':
        text += (f" If {poss} income rises by {{pct}}%, by what percentage does {poss} quantity of {st['gx']} "
                 f"demanded change? Round to two decimals.")
    else:
        text += (f" {poss[0].upper() + poss[1:]} quantity of {st['gx']} demanded changed by {{pct}}%. By what "
                 f"percentage did {poss} income change? Round to two decimals.")
    return (text,)


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, st in enumerate(STORIES):
        for tag in PERSPECTIVES:
            for qtype in TYPES:
                out.append(ct.CodeTemplate(parts=_parts(st, tag, k, qtype), type=qtype, domain=st['domain'],
                                           tags=tag, difficulty_level=1, spec={'story': k, 'type': qtype}))
    return out


def pct(x: float) -> str:
    s = num(x)
    return ('+' + s if not s.startswith('-') and s != '0' else s) + '%'


def sign_balanced(r, key, cands, shown):
    """Three distractors from cands (sign flips of the key left out of cands): in half the draws none is a sign flip
    of another option; in the other half the options are two +/- pairs, the key with -key and one mistake with its
    sign flip. Review, round 3: -key was the only +/- pair, which narrowed the choice to two."""
    if r.random() < 0.5:
        for _ in range(20):
            dist = pick_apart(r, key, cands, lo=-1000, hi=1000, shown=shown)
            if dist is None:
                return None
            vals = [shown(z) for z in [key] + dist]
            if not any(shown(-z) in vals for z in [key] + dist):
                return dist
        return None
    one = pick_apart(r, key, [c for c in cands if abs(abs(c) - abs(key)) >= 0.1 * abs(key)], n=1, lo=-1000,
                     hi=1000, shown=shown)
    if one is None:
        return None
    out = [-key, one[0], -one[0]]
    vals = [shown(z) for z in [key] + out]
    return out if len(set(vals)) == 4 else None


def draw(r, qtype: str) -> dict:
    for _ in range(100000):
        if qtype == 'direct':
            m1 = 50 * int(r.integers(4, 61))
            m2 = m1 + 50 * int(r.integers(1, 21)) * (1 if r.random() < 0.85 else -1)
            if m2 <= 0:
                continue
            q1 = int(r.integers(4, 121))
            target = float(r.uniform(0.2, 2.5)) * (1 if r.random() < 0.75 else -1)
            q2 = round(q1 * (1 + target * (m2 - m1) / m1))
            if q2 <= 0 or q2 == q1:
                continue
            dq, dm = (q2 - q1) / q1, (m2 - m1) / m1
            key = dq / dm
            mid = ((q2 - q1) / ((q1 + q2) / 2)) / ((m2 - m1) / ((m1 + m2) / 2))
            cands = [dm / dq, mid, (q2 - q1) / (m2 - m1), ((q2 - q1) / q2) / ((m2 - m1) / m2)]
            dist = sign_balanced(r, key, cands, num)
            if dist is None or abs(key) < 0.05 or near_tie(key):
                continue
            return {'income_start': f'{m1:,}', 'income_end': f'{m2:,}', 'quantity_start': str(q1),
                    'quantity_end': str(q2), 'options_list': [[num(z) for z in [key] + dist]],
                    'correct_answer': [[1, 0, 0, 0]]}
        E = round(float(r.uniform(0.2, 3.0)), 2) * (1 if r.random() < 0.75 else -1)
        k = round(float(r.uniform(2, 25)) * 2) / 2 * (1 if qtype == 'quantity' or r.random() < 0.7 else -1)
        if qtype == 'quantity':
            key = E * k
            cands = [k / E, E + k, E * k / 100]
        else:
            key = k / E
            cands = [E * k, k - E, E / k]
        dist = sign_balanced(r, key, cands, pct)
        if dist is None or abs(key) < 0.1 or near_tie(key):   # 1.31 x 24.5 = 32.095
            continue
        return {'elasticity': num(E), 'pct': num(k), 'options_list': [[pct(z) for z in [key] + dist]],
                'correct_answer': [[1, 0, 0, 0]]}
    raise RuntimeError(f'income_elasticity: no draw for {qtype}')


@function_registry.register(ELEMENT)
def fill_income_elasticity(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    return draw(filler_utils.rng, spec['type'])


ct.register(ELEMENT, _templates())

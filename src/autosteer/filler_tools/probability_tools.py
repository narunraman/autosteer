"""Fillers for the v1-era STEER probability elements gamblers_fallacy and bayes_rule (recovered 2026-10 from the
rationality_benchmark fill notebooks; templates: STEER-generation's audit/steer/regen/recover_probability_templates.py).

gamblers_fallacy  fill_generations/gamblers_fallacy.ipynb: template x streak length; options Yes / No. Every
                  template asks whether a run of independent outcomes makes some outcome more (or less) likely next
                  time, so the key is "No". Streak words as in the notebook, without "one" ("... the last one
                  rounds" is ungrammatical in most templates): three, six, seven, ten, twelve.
bayes_rule        fill_generations/bayes_rule.ipynb (bayes_rule_q_gen): integer percentages P(A), P(B), P(B|A) in
                  1-99; the answer P(A|B) = P(B|A) P(A) / P(B), shown with 2 decimals. The notebook's draw could be
                  incoherent; draws are kept only if P(A and B) <= min(P(A), P(B)) and P(B and not A) <= P(not A),
                  and the answer lies in [1%, 99%]. Options (re-derived; the original distractors are lost and
                  were sometimes negative): the answer and two of the usual mistakes, P(B|A) (inverting the
                  conditional), P(A) (base rate), P(B|A) P(B) / P(A) (wrong inversion), P(B|A) P(A) (the joint
                  probability) and P(A) / P(B) (likelihood left out), each at least max(1 point, 5%) from the
                  others, at most one of them a number copied from the question; all with 2 decimals, and the
                  question asks for 2 decimals.
"""
from __future__ import annotations

from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()
STREAKS = ['three', 'six', 'seven', 'ten', 'twelve']


def _rng():
    return filler_utils.rng


@function_registry.register('gamblers_fallacy')
def fill_gamblers_fallacy(metadata: dict) -> dict:
    return {'streak': STREAKS[int(_rng().integers(len(STREAKS)))],
            'options_list': [['Yes', 'No']], 'correct_answer': [[0, 1]]}


def _apart(v, vals):
    return all(abs(v - w) >= max(1.0, 0.05 * max(v, w)) for w in vals)


# The question states the rounding (BAYES_NOTE, stated in the stored templates), and every option
# has two decimals. At most one distractor is a number copied from the question (P(B|A) or P(A)); the other is a
# computed mistake (the wrong inversion P(B|A) P(B) / P(A), the joint P(B|A) P(A), or P(A) / P(B), which leaves out
# the likelihood), so the key is not singled out as the only option that needs a computation.
BAYES_NOTE = ' Give the answer as a percentage rounded to two decimal places.'


@function_registry.register('bayes_rule')
def fill_bayes_rule(metadata: dict) -> dict:
    while True:
        PA, PB, PBA = (int(_rng().integers(1, 100)) for _ in range(3))
        joint = PBA * PA / 100
        ans = PBA * PA / PB
        if joint > min(PA, PB) or PB - joint > 100 - PA or not 1 <= ans <= 99:
            continue
        given = [float(PBA), float(PA)]
        computed = [PBA * PB / PA, joint, 100 * PA / PB]
        cands = [(c, True) for c in given] + [(c, False) for c in computed]
        vals, copied = [round(ans, 2)], 0
        for i in _rng().permutation(len(cands)):
            c, is_given = cands[int(i)]
            c = round(c, 2)
            if len(vals) < 3 and 0 < c <= 100 and _apart(c, vals) and not (is_given and copied):
                vals.append(c)
                copied += is_given
        if len(vals) == 3:
            break
    return {'PA': PA, 'PB': PB, 'PBA': PBA, 'options_list': [[f'{v:.2f}%' for v in vals]],
            'correct_answer': [[1, 0, 0]]}


# ------------------------------------------------------------------ complement_rule
# rationality_benchmark fill_generations/complement.ipynb: PA = round(random(), 2) and, for two-probability
# templates (difficulty 1), PB uniform in [0, 1 - PA]; the answer is 1 - PA (- PB). Here PA, PB >= 0.01 and the
# answer >= 0.02. Three options (as released): the answer and two nearby probabilities, drawn from the slips
# 1 - PA and 1 - PB (forgetting one of the two given events, difficulty 1), PA itself, and the answer +- 0.03-0.15;
# all in [0.01, 0.99], at least 0.03 apart.
@function_registry.register('complement_rule')
def fill_complement_rule(metadata: dict) -> dict:
    hard = int(metadata.get('difficulty_level', 0)) >= 1
    while True:
        PA = round(float(_rng().uniform(0.01, 0.97)), 2)
        PB = round(float(_rng().uniform(0.01, 1 - PA)), 2) if hard else 0.0
        ans = round(1 - PA - PB, 2)
        if ans < 0.02 or (hard and PB < 0.01):
            continue
        cands = [PA] + ([round(1 - PA, 2), round(1 - PB, 2)] if hard else [])
        cands += [round(ans + float(_rng().choice([-1, 1])) * float(_rng().uniform(0.03, 0.15)), 2) for _ in range(4)]
        vals = [ans]
        for i in _rng().permutation(len(cands)):
            c = cands[int(i)]
            if len(vals) < 3 and 0.01 <= c <= 0.99 and all(abs(c - v) >= 0.03 - 1e-9 for v in vals):
                vals.append(c)
        if len(vals) == 3:
            break
    out = {'PA': PA, 'options_list': [[str(v) for v in vals]], 'correct_answer': [[1, 0, 0]]}
    if hard:
        out['PB'] = PB
    else:
        out['PB'] = 0.5  # unused by one-probability templates; keeps the key set the same for the sampler
    return out

"""loss_aversion (STEER 2.4.e, Avoidance of Loss Averse Behavior): hand-written templates.

Design of 6 October 2026 (no earlier generator; the 2023 choices13k proof of concept is not used). A mixed gamble
(win $G with probability p, lose $L otherwise) whose expected value is positive, framed so that loss aversion pulls
toward rejecting it: the loss is spelled out as the decision maker's own money, and the probability-weighted gain is
only 1.1 to 2 times the probability-weighted loss (inside the range a loss-aversion coefficient of about 2 would
reject). Every question says the decision maker faces many small decisions like this and the amounts are small
compared with their savings and that they care about the average result of such decisions, so the expected value is
the criterion (review, October 2026: "small compared with savings" alone left risk aversion as a
defensible reason to decline). Answer-graded: the key is the option with the higher expected value.

Types:
    accept_reject (difficulty 1): the gamble or nothing. In two thirds of the questions the expected value is
        positive (at least 10% of the probability-weighted loss; the key is to take the gamble); in one third it is
        negative (probability-weighted gain 0.5 to 0.87 times the probability-weighted loss; the key is to decline),
        so an "always take the gamble" rule fails (before October 2026 it was always positive).
    sure_amount (difficulty 2): the gamble or a small sure payment S; the gamble is better in two thirds of the
        questions (S at most EV / 1.15), the sure payment in one third (S at least 1.15 EV), so a "always gamble"
        rule fails.
Domains (stories): a coin-flip bet (p = 1/2), a marketing campaign, a short-term trade, a game-show wheel, a bonus /
deduction pay scheme; probabilities 25% to 80% outside the coin flip. Perspectives: first_person, second_person,
man, woman, anon, ai_assistant (advising a client).
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import possessive_pronouns, voice

function_registry = FillRegister()

ELEMENT = 'loss_aversion'
PERSPECTIVES = ('first_person', 'second_person', 'man', 'woman', 'anon', 'ai_assistant')
# domain, anon subject, story (voice slots in [brackets]), accept option, reject option
STORIES = [
    ('gambling', 'the player',
     "A friend offers [obj] a bet on one flip of a fair coin: heads, [subj] win[s] ${G}; tails, [subj] pay[s] the "
     "friend ${L} out of [poss] own pocket.", 'Take the bet', 'Decline the bet'),
    ('business', 'the shop owner',
     "[Subj] can run a small marketing campaign for [poss] shop. There is {p} chance that it brings in ${G} of extra "
     "profit; otherwise it flops and [subj] lose[s] the ${L} it cost.", 'Run the campaign', 'Do not run the campaign'),
    ('investing', 'the investor',
     "[Subj] can make a short-term trade that gains ${G} with {p} chance and otherwise loses ${L} of [poss] money.",
     'Make the trade', 'Do not make the trade'),
    ('game_show', 'the contestant',
     "On a game show, [subj] [have] won some money and can now spin a wheel: with {p} chance [subj] win[s] another "
     "${G}, and otherwise [subj] must hand back ${L} of what [subj] already won.", 'Spin the wheel',
     'Do not spin the wheel'),
    ('workplace', 'the engineer',
     "[Poss] employer offers an optional pay scheme for the next project: a bonus of ${G} if the project finishes on "
     "time, which has {p} chance, and a deduction of ${L} from [poss] pay if it does not.", 'Join the scheme',
     'Stay on the standard pay'),
]
CRITERION = ("[Subj] face[s] many small decisions like this one, the amounts are small compared with [poss] "
             "savings, and [subj] care[s] about the average result of such decisions over time.")
P_CHOICES = [(1, 4), (3, 10), (2, 5), (1, 2), (3, 5), (7, 10), (3, 4), (4, 5)]


def _fill_voice(text: str, v: dict) -> str:
    for k in ('Subj', 'subj', 'Poss', 'poss', 'have', 's', 'q', 'obj'):
        text = text.replace(f'[{k}]', v[k])
    return text


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for d, (dom, anon_subj, story, acc, rej) in enumerate(STORIES):
        for k, persp in enumerate(PERSPECTIVES):
            if persp == 'ai_assistant':
                v = voice('anon', k, 'your client', "your client's")
                lead = 'You are an AI assistant advising a client. '
            else:
                v = voice(persp, d + k, anon_subj, None)
                lead = ''
            if v['subj'] == 'you':
                v['s'] = ''
            anon = {v['subj']: 'their'} if persp == 'anon' else None   # "out of the player's own pocket"
            body = lead + possessive_pronouns(_fill_voice(story, v), anon) + ' '
            crit = possessive_pronouns(_fill_voice(CRITERION, v), anon)
            crit = crit[0].upper() + crit[1:]
            q = f"What should {v['q']} do?"
            for t in ('accept_reject', 'sure_amount'):
                alt = (' Alternatively, ' + v['subj'] + ' can take a guaranteed ${S} instead.') if t == 'sure_amount' \
                    else ''
                text = body.rstrip() + alt + ' ' + crit + ' ' + q
                text = text[0].upper() + text[1:]
                out.append(ct.CodeTemplate(parts=(text,), type=t, domain=dom, tags=persp,
                                           difficulty_level=1 if t == 'accept_reject' else 2,
                                           spec={'story': d, 'type': t}))
    return out


def _p_text(p) -> str:
    return f'a {round(100 * p[0] / p[1])}%'


@function_registry.register(ELEMENT)
def fill_loss_aversion(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    dom, _, _, acc, rej = STORIES[spec['story']]
    r = filler_utils.rng
    negative = spec['type'] == 'accept_reject' and bool(r.random() < 1 / 3)
    lo_r, hi_r = (0.5, 0.87) if negative else (1.1, 2.0)
    while True:
        p = (1, 2) if dom == 'gambling' else P_CHOICES[int(r.integers(len(P_CHOICES)))]
        pw, pl = p[0] / p[1], 1 - p[0] / p[1]
        L = 5 * int(r.integers(4, 81))
        ratio = float(r.uniform(lo_r, hi_r))  # probability-weighted gain / probability-weighted loss
        G = 5 * round(ratio * pl * L / pw / 5)
        ev = pw * G - pl * L
        if G <= 0 or G == L or not (lo_r <= pw * G / (pl * L) <= hi_r) or abs(ev) < max(1.0, 0.1 * pl * L):
            continue
        out = {'G': f'{G:,}', 'L': f'{L:,}', 'p': _p_text(p)}
        if spec['type'] == 'accept_reject':
            options, key = [acc, rej], (rej if negative else acc)
        else:
            sure_better = bool(r.random() < 1 / 3)
            if sure_better:
                lo, hi = 1.15 * ev, min(2.0 * ev, 0.9 * G)
            else:
                lo, hi = 0.3 * ev, ev / 1.15
            cands = [s for s in range(5 * int(lo // 5) + 5, int(hi) + 1, 5) if lo <= s <= hi]
            if not cands:
                continue
            S = cands[int(r.integers(len(cands)))]
            out['S'] = f'{S:,}'
            sure = f'Take the guaranteed ${S:,}'
            options, key = [acc, sure], (sure if sure_better else acc)
        out['options_list'] = [options]
        out['correct_answer'] = [[int(o == key) for o in options]]
        return out


ct.register(ELEMENT, _templates())

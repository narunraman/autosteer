"""Build the reflection_effect template dataset (gain / loss mirror pairs, STEER Element 2.5.c).

    uv run python data/templates/reflection_effect/build_templates.py      # from the repo root

Writes dataset.pkl (+ legacy pickles) next to this file. Deterministic (fixed seed); no LLM involved.

Every template is one two-part item in a single story:
  sub_id 0 (gains):   option 0 = gain X for sure    option 1 = gain Y with probability p, else nothing
  sub_id 1 (losses):  option 0 = lose X for sure    option 1 = lose Y with probability p, else nothing
The filler (single_agent_tools.gen_reflection_effect) supplies {x_amount} {y_amount} {p_pct} {p_art} with
p*Y = X exactly. Option 0 is always the sure side. Type two_part: no option is marked correct (consistency-
graded, rule reflection_same_side in src/release/consistency.py); both parts are asked fresh. Type
stated_prior: one question stating the gain-frame choice (half sure, half lottery) and asking the loss-frame
decision; the same-side option is marked correct.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)

from src.filler_tools.utilities.risk_pair_templates import (  # noqa: E402
    Item, Part, build_dataset, fill_context, now_context, opt, stated_prior_items,
)

SEED = 20260929
DIFFICULTY = 7  # same grade level as certainty_effect (the released STEER grade of the risk-bias elements)


def F(what, ctx, sure, lot):
    """One frame (gain or loss) of a scenario."""
    return dict(what=what, ctx=ctx, sure=sure, lot=lot)


S = dict
SCENARIOS = [
    # ------------------------------------------------------------------ finance
    S(domain='finance', type='money', role='investor',
      gain=F(('investment', 'investments'),
             ['%(S)s %(have)s some savings to invest for one year, and %(pos)s bank offers two investment products.',
              '%(S)s %(be)s deciding how to invest a year-end bonus and %(have)s narrowed it down to two '
              'investments.'],
             'An investment with a guaranteed profit of $AMT',
             'An investment with a PROB chance of a profit of $AMT, and no profit otherwise'),
      loss=F(('exit', 'exits'),
             ['%(S)s %(have)s an investment that has gone badly, and %(pos)s bank offers two ways to close it out.',
              '%(S)s must unwind a losing investment by the end of the year and %(have)s narrowed it down to two '
              'exits.'],
             'An exit with a certain loss of $AMT',
             'An exit with a PROB chance of a loss of $AMT, and no loss otherwise')),
    S(domain='finance', type='money', role='business owner',
      gain=F(('contract', 'contracts'),
             ['%(S)s %(be)s choosing between two client contracts for %(pos)s company next quarter.',
              '%(S)s %(be)s running a small design studio and can accept only one of two contracts offered this '
              'month.'],
             'A fixed-fee contract that pays $AMT for certain',
             'A performance contract with a PROB chance of paying $AMT, and nothing otherwise'),
      loss=F(('option', 'options'),
             ['%(S)s %(be)s being sued by a client over a failed project, and %(pos)s lawyer lays out two options.',
              '%(S)s %(be)s facing a penalty claim from a client over a late delivery and can resolve it in one of '
              'two ways.'],
             'Settle and pay $AMT for certain',
             'Go to court with a PROB chance of paying $AMT, and paying nothing otherwise')),
    S(domain='finance', type='money', role='claimant',
      gain=F(('resolution route', 'resolution routes'),
             ['%(S)s %(be)s owed money after a contract dispute, and %(pos)s lawyer lays out two ways to resolve '
              'it.',
              '%(S)s %(have)s filed a claim against a supplier, and a mediator proposes two ways to settle it.'],
             'A route that recovers $AMT for certain',
             'A route with a PROB chance of recovering $AMT, and nothing otherwise'),
      loss=F(('resolution route', 'resolution routes'),
             ['%(S)s %(have)s been accused of breaching a contract, and %(pos)s lawyer lays out two ways to resolve '
              'the claim.',
              '%(S)s %(be)s the target of a claim from a former supplier, and a mediator proposes two ways to settle '
              'it.'],
             'A route that costs $AMT for certain',
             'A route with a PROB chance of costing $AMT, and costing nothing otherwise')),
    S(domain='finance', type='money', role='trader',
      gain=F(('trade', 'trades'),
             ['%(S)s %(have)s enough capital for only one of two short-term trades this week.',
              '%(S)s %(be)s reviewing two option trades that expire next month and can place only one of them.'],
             'A trade that locks in a gain of $AMT for certain',
             'A trade with a PROB chance of a gain of $AMT, and no gain otherwise'),
      loss=F(('exit strategy', 'exit strategies'),
             ['%(S)s %(be)s holding a losing position that must be closed this week with one of two exit '
              'strategies.',
              '%(S)s %(have)s an option trade that has moved against %(obj)s and can close it with one of two exit '
              'strategies.'],
             'A strategy that locks in a loss of $AMT for certain',
             'A strategy with a PROB chance of a loss of $AMT, and no loss otherwise')),
    # ------------------------------------------------------------------ medical
    S(domain='medical', type='count', role='hospital director',
      gain=F(('program', 'programs'),
             ["%(S)s %(be)s in charge of a hospital's budget for a new treatment program and can fund only one of "
              'two programs.',
              '%(S)s %(be)s deciding which of two treatment programs a regional hospital should run next year.'],
             'A program that is certain to save AMT patients',
             'A program with a PROB chance of saving AMT patients, and saving no one otherwise'),
      loss=F(('rationing program', 'rationing programs'),
             ["%(S)s %(be)s in charge of a hospital's response to a drug shortage and must choose one of two "
              'rationing programs.',
              '%(S)s %(be)s deciding how a regional hospital will ration a scarce drug next year.'],
             'A program under which AMT patients will certainly die',
             'A program with a PROB chance that AMT patients die, and no deaths otherwise')),
    S(domain='medical', type='count', role='public health official',
      gain=F(('plan', 'plans'),
             ['%(S)s %(be)s planning a vaccination campaign during an outbreak and can adopt one of two plans.',
              '%(S)s %(be)s responsible for protecting a city during a flu outbreak and must choose between two '
              'vaccination plans.'],
             'A plan that is certain to protect AMT people',
             'A plan with a PROB chance of protecting AMT people, and protecting no one otherwise'),
      loss=F(('containment plan', 'containment plans'),
             ['%(S)s %(be)s managing an outbreak that is spreading quickly and must adopt one of two containment '
              'plans.',
              '%(S)s %(be)s responsible for a city during a flu outbreak and must choose between two containment '
              'plans.'],
             'A plan under which AMT people will certainly be infected',
             'A plan with a PROB chance that AMT people are infected, and no infections otherwise')),
    S(domain='medical', type='count', role='clinic manager',
      gain=F(('screening program', 'screening programs'),
             ['%(S)s %(be)s choosing a cancer screening program for a rural clinic.',
              '%(S)s %(have)s funding for one of two screening programs at a community clinic.'],
             'A program that is certain to detect AMT early-stage cases',
             'A program with a PROB chance of detecting AMT early-stage cases, and none otherwise'),
      loss=F(('reduced program', 'reduced programs'),
             ["%(S)s %(be)s cutting a rural clinic's cancer screening budget and must choose one of two reduced "
              'programs.',
              '%(S)s %(have)s to scale back the screening programs at a community clinic in one of two ways.'],
             'A program under which AMT early-stage cases will certainly go undetected',
             'A program with a PROB chance that AMT early-stage cases go undetected, and none otherwise')),
    # ------------------------------------------------------------------ employment
    S(domain='employment', type='money', role='job candidate',
      gain=F(('job offer', 'job offers'),
             ['%(S)s %(have)s two job offers that are identical except for the signing bonus.',
              '%(S)s %(be)s about to accept a new job, and the employer lets new hires pick one of two signing '
              'bonuses.'],
             'An offer with a guaranteed signing bonus of $AMT',
             'An offer with a PROB chance of a signing bonus of $AMT, and no bonus otherwise'),
      loss=F(('arrangement', 'arrangements'),
             ['%(S)s %(be)s leaving a job early and must repay part of a signing bonus under one of two '
              'arrangements.',
              '%(S)s %(be)s breaking a job contract, and HR offers two arrangements for settling the early-exit '
              'penalty.'],
             'An arrangement with a certain repayment of $AMT',
             'An arrangement with a PROB chance of a repayment of $AMT, and no repayment otherwise')),
    S(domain='employment', type='money', role='freelancer',
      gain=F(('payment arrangement', 'payment arrangements'),
             ['%(S)s %(have)s been asked to do a one-off freelance project, and the client proposes two ways of '
              'paying.',
              '%(S)s %(be)s negotiating the fee for a short consulting job and %(have)s two payment arrangements to '
              'choose from.'],
             'A flat fee of $AMT, paid for certain',
             'A success fee with a PROB chance of paying $AMT, and nothing otherwise'),
      loss=F(('penalty arrangement', 'penalty arrangements'),
             ['%(S)s delivered a freelance project late, and the client proposes two ways of handling the penalty.',
              '%(S)s %(be)s negotiating with a client over a missed deadline and %(have)s two penalty arrangements '
              'to choose from.'],
             'A flat penalty of $AMT, paid for certain',
             'A penalty with a PROB chance of costing $AMT, and nothing otherwise')),
    S(domain='employment', type='money', role='sales representative',
      gain=F(('bonus plan', 'bonus plans'),
             ['%(S)s %(be)s choosing a year-end bonus plan offered by %(pos)s employer.',
              "%(S)s must pick one of two bonus plans in %(pos)s company's new compensation scheme."],
             'A plan that pays a bonus of $AMT for certain',
             'A plan with a PROB chance of paying a bonus of $AMT, and no bonus otherwise'),
      loss=F(('deduction plan', 'deduction plans'),
             ['%(S)s missed a sales target, and %(pos)s employer offers two ways to claw back an advance on '
              'commission.',
              '%(S)s must pick one of two plans for how a shortfall in %(pos)s sales quota is deducted from pay.'],
             'A plan that deducts $AMT for certain',
             'A plan with a PROB chance of deducting $AMT, and no deduction otherwise')),
    # ------------------------------------------------------------------ gambling
    S(domain='gambling', type='money', role='contestant',
      gain=F(('choice', 'choices'),
             ['%(S)s %(have)s reached the final round of a TV game show, and the host offers two choices.',
              '%(S)s %(be)s on a radio quiz show and, after answering the last question, can pick one of two '
              'prizes.'],
             'Take $AMT in cash for certain',
             'Spin a wheel with a PROB chance of winning $AMT, and nothing otherwise'),
      loss=F(('choice', 'choices'),
             ['%(S)s %(have)s reached the penalty round of a TV game show, where contestants give back part of '
              'their winnings, and the host offers two choices.',
              '%(S)s %(be)s on a radio quiz show and, after missing the last question, must pick one of two '
              'penalties.'],
             'Give back $AMT of the winnings for certain',
             'Spin a wheel with a PROB chance of giving back $AMT, and nothing otherwise')),
    S(domain='gambling', type='money', role='casino visitor',
      gain=F(('free bet', 'free bets'),
             ['%(S)s %(be)s at a casino that offers new members one free bet out of two.',
              '%(S)s %(have)s a voucher for one free bet at a casino and %(be)s choosing between two bets.'],
             'A free bet that pays $AMT for certain',
             'A free bet with a PROB chance of paying $AMT, and nothing otherwise'),
      loss=F(('settlement', 'settlements'),
             ['%(S)s %(have)s a losing hand at a casino table and must choose between two ways to settle it.',
              '%(S)s %(be)s at a casino and %(have)s to cover a lost wager with one of two settlements.'],
             'Pay $AMT for certain',
             'A final draw with a PROB chance of paying $AMT, and paying nothing otherwise')),
    S(domain='gambling', type='money', role='player',
      gain=F(('deck', 'decks'),
             ['%(S)s %(be)s playing a promotional card game at a fair and can draw one card from one of two decks.',
              'At a street festival, %(s1)s can draw a single card from one of two prize decks.'],
             'A deck in which every card wins $AMT',
             'A deck with a PROB chance of drawing a card that wins $AMT, and nothing otherwise'),
      loss=F(('deck', 'decks'),
             ['%(S)s %(be)s playing a card game at a fair in which the loser draws one card from one of two penalty '
              'decks.',
              'At a street festival, %(s1)s lost a round of a card game and must draw a single card from one of two '
              'penalty decks.'],
             'A deck in which every card costs $AMT',
             'A deck with a PROB chance of drawing a card that costs $AMT, and nothing otherwise')),
]

# Few-shot pool (disjoint stories), one scenario per domain.
EXAMPLE_SCENARIOS = [
    S(domain='finance', type='money', role='retiree',
      gain=F(('certificate', 'certificates'),
             ['%(S)s %(have)s some spare cash and %(be)s comparing two one-year certificates from %(pos)s credit '
              'union.',
              '%(S)s %(be)s moving part of a pension pot into one of two one-year certificates.'],
             'A certificate paying interest of $AMT for certain',
             'A market-linked certificate with a PROB chance of paying $AMT, and nothing otherwise'),
      loss=F(('withdrawal term', 'withdrawal terms'),
             ['%(S)s must cash out a certificate early and can choose between two early-withdrawal terms.',
              '%(S)s %(be)s withdrawing part of a pension pot early and must pick one of two withdrawal terms.'],
             'A term with a certain penalty of $AMT',
             'A term with a PROB chance of a penalty of $AMT, and no penalty otherwise')),
    S(domain='medical', type='count', role='health minister',
      gain=F(('strategy', 'strategies'),
             ['%(S)s %(be)s allocating a limited supply of a new medicine and must choose between two treatment '
              'strategies.',
              '%(S)s %(be)s choosing how a national health service will deploy a scarce new drug.'],
             'A strategy that is certain to cure AMT patients',
             'A strategy with a PROB chance of curing AMT patients, and no cures otherwise'),
      loss=F(('strategy', 'strategies'),
             ['%(S)s %(be)s rationing a scarce medicine after a supply failure and must choose between two '
              'strategies.',
              '%(S)s %(be)s choosing how a national health service will cope with a shortage of a new drug.'],
             'A strategy under which AMT patients will certainly go without treatment',
             'A strategy with a PROB chance that AMT patients go without treatment, and no one otherwise')),
    S(domain='employment', type='money', role='consultant',
      gain=F(('payment term', 'payment terms'),
             ['%(S)s %(be)s negotiating a consulting contract and can choose between two payment terms.',
              '%(S)s %(have)s been hired for a short advisory project and %(be)s offered two ways to be paid.'],
             'A fixed payment of $AMT, guaranteed',
             'A contingent payment with a PROB chance of paying $AMT, and nothing otherwise'),
      loss=F(('refund term', 'refund terms'),
             ['%(S)s %(be)s disputing a consulting invoice and can choose between two refund terms.',
              '%(S)s missed a milestone on an advisory project and %(be)s offered two refund terms for the client.'],
             'A fixed refund of $AMT, paid for certain',
             'A contingent refund with a PROB chance of costing $AMT, and nothing otherwise')),
    S(domain='gambling', type='money', role='shopper',
      gain=F(('panel', 'panels'),
             ['%(S)s %(have)s a promotional scratch card with two panels and may scratch only one.',
              'After a purchase, %(s1)s receives a scratch card and may scratch one of its two panels.'],
             'A panel that reveals a prize of $AMT for certain',
             'A panel with a PROB chance of revealing a prize of $AMT, and nothing otherwise'),
      loss=F(('panel', 'panels'),
             ['%(S)s %(have)s a forfeit scratch card from a lost bet and may scratch only one of its two panels.',
              'After losing a bet with friends, %(s1)s must scratch one of the two panels on a forfeit card.'],
             'A panel that reveals a charge of $AMT for certain',
             'A panel with a PROB chance of revealing a charge of $AMT, and nothing otherwise')),
]

ASK = [  # (part 0, part 1): different wording, same meaning; neither mentions the other part
    ('Which %(what)s %(Q)s choose?', 'Which of the following %(whats)s %(Q)s choose?'),
    ('Which of the two %(whats)s %(Q)s take?', 'Which %(what)s %(Q)s pick?'),
]


def frame_part(frame: dict, ci: int, tok: dict, ask: str) -> Part:
    """One frame as a question part: sure X vs (Y, p). Sure side first."""
    tok = dict(tok, what=frame['what'][0], whats=frame['what'][1])
    return Part(f"{fill_context(frame['ctx'][ci], tok)} {ask % tok}",
                [opt(frame['sure'], None, 'x_amount'), opt(frame['lot'], 'p_pct', 'y_amount')])


def render(sc: dict, ci: int, tok: dict, rng) -> list[Item]:
    """two_part: part 0 = gain frame, part 1 = loss frame (same story, same X, Y, p).
    stated_prior (x2): states the gain-frame choice, asks the loss-frame decision (sure -> sure, risky -> risky)."""
    ask0, ask1 = ASK[rng.randrange(len(ASK))]
    gain, loss = sc['gain'], sc['loss']
    p0, p1 = frame_part(gain, ci, tok, ask0), frame_part(loss, ci, tok, ask1)
    ltok = dict(tok, what=loss['what'][0], whats=loss['what'][1])
    stated = stated_prior_items(p0, now_context(loss['ctx'][ci], ltok, sc['role']), ask1 % ltok, p1, tok,
                                gain['what'][1])
    return [Item('two_part', '', [p0, p1])] + stated


def main():
    for sc in SCENARIOS + EXAMPLE_SCENARIOS:
        sc['ctx'] = sc['gain']['ctx']  # build_frames iterates over len(sc['ctx'])
        assert len(sc['gain']['ctx']) == len(sc['loss']['ctx'])
    build_dataset('reflection_effect', HERE, SCENARIOS, EXAMPLE_SCENARIOS, render, SEED, DIFFICULTY)


if __name__ == '__main__':
    main()

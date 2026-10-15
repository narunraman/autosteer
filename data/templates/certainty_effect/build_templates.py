"""Build the certainty_effect template dataset (Allais / common-ratio pairs).

    uv run python data/templates/certainty_effect/build_templates.py      # from the repo root

Writes dataset.pkl (+ the legacy questions/options/answers/questions_metadata pickles) next to this file.
Deterministic (fixed seed); no LLM involved.

Every template is one two-part item in a single story:
  sub_id 0:  option 0 = X for sure              option 1 = Y with probability p, else nothing
  sub_id 1:  option 0 = X with probability k    option 1 = Y with probability k*p, else nothing
Option 0 is always the sure side (A / C) and option 1 the risky side (B / D). Type two_part: no option is
marked correct (consistency-graded, see src/release/consistency.py). Type stated_prior: one question stating
the part-0 choice (half the templates state A, half B) and asking the part-1 decision; the EU-implied option
(A -> C, B -> D) is marked correct. The filler (single_agent_tools.gen_certainty_effect)
supplies {x_amount} {y_amount} {p_pct} {k_pct} {kp_pct} and the articles {p_art} {k_art} {kp_art}. Both parts are self-contained questions: part 1 does
not state or refer to the answer given in part 0.
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

SEED = 20260928
DIFFICULTY = 7  # the released STEER grade level of this element (all 8,030 released rows were grade 7)

# Each scenario: domain, filler type ('money' | 'count'), anonymous role, option noun (singular, plural),
# two contexts, and the sure / lottery option patterns (AMT = amount, PROB = probability).
S = dict
SCENARIOS = [
    # ------------------------------------------------------------------ finance
    S(domain='finance', type='money', role='investor', what=('investment', 'investments'),
      ctx=['%(S)s %(have)s some savings to invest for one year, and %(pos)s bank offers two investment products.',
           '%(S)s %(be)s deciding how to invest a year-end bonus for the coming year and %(have)s narrowed it down '
           'to two investments.'],
      sure='An investment with a guaranteed profit of $AMT',
      lot='An investment with a PROB chance of a profit of $AMT, and no profit otherwise'),
    S(domain='finance', type='money', role='business owner', what=('contract', 'contracts'),
      ctx=['%(S)s %(be)s choosing between two client contracts for %(pos)s company next quarter.',
           '%(S)s %(be)s running a small design studio and can accept only one of two contracts offered this month.'],
      sure='A fixed-fee contract that pays $AMT for certain',
      lot='A performance contract with a PROB chance of paying $AMT, and nothing otherwise'),
    S(domain='finance', type='money', role='entrepreneur', what=('offer', 'offers'),
      ctx=['%(S)s %(be)s selling a small mobile app and %(have)s received two offers from buyers.',
           '%(S)s %(be)s ready to sell a side business, and two buyers have made offers.'],
      sure='An all-cash offer of $AMT, paid for certain',
      lot='An earn-out offer with a PROB chance of paying $AMT, and nothing otherwise'),
    S(domain='finance', type='money', role='shareholder', what=('payout option', 'payout options'),
      ctx=['%(S)s %(have)s shares in a company that is being restructured, and %(pos)s broker describes two payout '
           'options.',
           '%(S)s %(have)s a small stake in a private company that is being sold, and the buyer offers existing '
           'owners two payout options.'],
      sure='A cash payout of $AMT, guaranteed',
      lot='A payout with a PROB chance of paying $AMT, and no payout otherwise'),
    S(domain='finance', type='money', role='claimant', what=('resolution route', 'resolution routes'),
      ctx=['%(S)s %(be)s owed money after a contract dispute, and %(pos)s lawyer lays out two ways to resolve it.',
           '%(S)s %(have)s filed a claim against a supplier, and a mediator proposes two ways to settle it.'],
      sure='A route that recovers $AMT for certain',
      lot='A route with a PROB chance of recovering $AMT, and nothing otherwise'),
    S(domain='finance', type='money', role='trader', what=('trade', 'trades'),
      ctx=['%(S)s %(have)s enough capital for only one of two short-term trades this week.',
           '%(S)s %(be)s reviewing two option trades that expire next month and can place only one of them.'],
      sure='A trade that locks in a gain of $AMT for certain',
      lot='A trade with a PROB chance of a gain of $AMT, and no gain otherwise'),
    S(domain='finance', type='money', role='saver', what=('savings promotion', 'savings promotions'),
      ctx=['%(S)s %(be)s comparing two one-year savings promotions from different banks.',
           '%(S)s %(be)s opening a new savings account, and the bank lets new customers pick one of two '
           'promotions.'],
      sure='A promotion that pays a bonus of $AMT for certain',
      lot='A prize-linked promotion with a PROB chance of a bonus of $AMT, and no bonus otherwise'),
    S(domain='finance', type='money', role='property investor', what=('property deal', 'property deals'),
      ctx=['%(S)s %(be)s choosing between two short-term property deals.',
           '%(S)s %(have)s been offered a share in one of two renovation projects that will be sold next year.'],
      sure='A deal with a guaranteed profit of $AMT',
      lot='A deal with a PROB chance of a profit of $AMT, and no profit otherwise'),
    # ------------------------------------------------------------------ medical
    S(domain='medical', type='count', role='hospital director', what=('program', 'programs'),
      ctx=["%(S)s %(be)s in charge of a hospital's budget for a new treatment program and can fund only one of two "
           'programs.',
           '%(S)s %(be)s deciding which of two treatment programs a regional hospital should run next year.'],
      sure='A program that is certain to cure AMT patients',
      lot='A program with a PROB chance of curing AMT patients, and no cures otherwise'),
    S(domain='medical', type='count', role='public health official', what=('plan', 'plans'),
      ctx=['%(S)s %(be)s planning a vaccination campaign during an outbreak and can adopt one of two plans.',
           '%(S)s %(be)s responsible for protecting a city during a flu outbreak and must choose between two '
           'vaccination plans.'],
      sure='A plan that is certain to protect AMT people',
      lot='A plan with a PROB chance of protecting AMT people, and protecting no one otherwise'),
    S(domain='medical', type='count', role='clinic manager', what=('screening program', 'screening programs'),
      ctx=['%(S)s %(be)s choosing a cancer screening program for a rural clinic.',
           '%(S)s %(have)s funding for one of two screening programs at a community clinic.'],
      sure='A program that is certain to detect AMT early-stage cases',
      lot='A program with a PROB chance of detecting AMT early-stage cases, and none otherwise'),
    S(domain='medical', type='count', role='emergency coordinator', what=('evacuation plan', 'evacuation plans'),
      ctx=['%(S)s %(be)s coordinating the response to a flood and must choose one of two evacuation plans.',
           '%(S)s %(be)s preparing for an approaching wildfire and can put only one of two evacuation plans into '
           'action.'],
      sure='A plan that is certain to evacuate AMT residents safely',
      lot='A plan with a PROB chance of evacuating AMT residents safely, and none otherwise'),
    S(domain='medical', type='count', role='medical director', what=('protocol', 'protocols'),
      ctx=['%(S)s %(be)s deciding which of two treatment protocols a hospital network should adopt for a severe '
           'infection.',
           '%(S)s %(be)s choosing how an intensive care unit will treat patients during a disease outbreak.'],
      sure='A protocol that is certain to save AMT lives',
      lot='A protocol with a PROB chance of saving AMT lives, and none otherwise'),
    # ------------------------------------------------------------------ employment
    S(domain='employment', type='money', role='job candidate', what=('job offer', 'job offers'),
      ctx=['%(S)s %(have)s two job offers that are identical except for the signing bonus.',
           '%(S)s %(be)s about to accept a new job, and the employer lets new hires pick one of two signing '
           'bonuses.'],
      sure='An offer with a guaranteed signing bonus of $AMT',
      lot='An offer with a PROB chance of a signing bonus of $AMT, and no bonus otherwise'),
    S(domain='employment', type='money', role='freelancer', what=('payment arrangement', 'payment arrangements'),
      ctx=['%(S)s %(have)s been asked to do a one-off freelance project, and the client proposes two ways of paying.',
           '%(S)s %(be)s negotiating the fee for a short consulting job and %(have)s two payment arrangements to '
           'choose from.'],
      sure='A flat fee of $AMT, paid for certain',
      lot='A success fee with a PROB chance of paying $AMT, and nothing otherwise'),
    S(domain='employment', type='money', role='sales representative', what=('bonus plan', 'bonus plans'),
      ctx=['%(S)s %(be)s choosing a year-end bonus plan offered by %(pos)s employer.',
           "%(S)s must pick one of two bonus plans in %(pos)s company's new compensation scheme."],
      sure='A plan that pays a bonus of $AMT for certain',
      lot='A plan with a PROB chance of paying a bonus of $AMT, and no bonus otherwise'),
    S(domain='employment', type='money', role='engineer', what=('retention package', 'retention packages'),
      ctx=['%(S)s %(have)s been asked by %(pos)s employer to choose a retention package for staying another year.',
           '%(S)s %(be)s considering staying at %(pos)s company for another year, and HR offers two retention '
           'packages.'],
      sure='A retention payment of $AMT, guaranteed',
      lot='A package with a PROB chance of a retention payment of $AMT, and no payment otherwise'),
    S(domain='employment', type='money', role='employee', what=('reward', 'rewards'),
      ctx=['%(S)s %(have)s volunteered for a special project at work and can choose how the extra work is rewarded.',
           '%(S)s %(have)s agreed to cover extra shifts during a busy season, and %(pos)s manager offers two '
           'rewards.'],
      sure='A project bonus of $AMT, paid for certain',
      lot='A reward with a PROB chance of a project bonus of $AMT, and no bonus otherwise'),
    # ------------------------------------------------------------------ gambling
    S(domain='gambling', type='money', role='contestant', what=('choice', 'choices'),
      ctx=['%(S)s %(have)s reached the final round of a TV game show, and the host offers two choices.',
           '%(S)s %(be)s on a radio quiz show and, after answering the last question, can pick one of two prizes.'],
      sure='Take $AMT in cash for certain',
      lot='Spin a wheel with a PROB chance of winning $AMT, and nothing otherwise'),
    S(domain='gambling', type='money', role='guest', what=('ticket', 'tickets'),
      ctx=['At a charity gala, %(s1)s can pick one of two prize tickets.',
           'At a company party, %(s1)s may take one of two raffle tickets from the prize table.'],
      sure='A ticket that wins $AMT for certain',
      lot='A ticket with a PROB chance of winning $AMT, and nothing otherwise'),
    S(domain='gambling', type='money', role='casino visitor', what=('free bet', 'free bets'),
      ctx=['%(S)s %(be)s at a casino that offers new members one free bet out of two.',
           '%(S)s %(have)s a voucher for one free bet at a casino and %(be)s choosing between two bets.'],
      sure='A free bet that pays $AMT for certain',
      lot='A free bet with a PROB chance of paying $AMT, and nothing otherwise'),
    S(domain='gambling', type='money', role='radio listener', what=('prize option', 'prize options'),
      ctx=['%(S)s %(have)s won a radio call-in contest and must choose how to collect the prize.',
           '%(S)s %(have)s been drawn as the winner of a local radio giveaway and can choose between two prize '
           'options.'],
      sure='A cash prize of $AMT, guaranteed',
      lot='A mystery envelope with a PROB chance of containing $AMT, and nothing otherwise'),
    S(domain='gambling', type='money', role='player', what=('deck', 'decks'),
      ctx=['%(S)s %(be)s playing a promotional card game at a fair and can draw one card from one of two decks.',
           'At a street festival, %(s1)s can draw a single card from one of two prize decks.'],
      sure='A deck in which every card wins $AMT',
      lot='A deck with a PROB chance of drawing a card that wins $AMT, and nothing otherwise'),
]

# Few-shot pool (disjoint stories), one scenario per domain.
EXAMPLE_SCENARIOS = [
    S(domain='finance', type='money', role='retiree', what=('certificate', 'certificates'),
      ctx=['%(S)s %(have)s some spare cash and %(be)s comparing two one-year certificates from %(pos)s credit union.',
           '%(S)s %(be)s moving part of a pension pot into one of two one-year certificates.'],
      sure='A certificate paying interest of $AMT for certain',
      lot='A market-linked certificate with a PROB chance of paying $AMT, and nothing otherwise'),
    S(domain='medical', type='count', role='health minister', what=('strategy', 'strategies'),
      ctx=['%(S)s %(be)s allocating a limited supply of a new medicine and must choose between two treatment '
           'strategies.',
           '%(S)s %(be)s choosing how a national health service will deploy a scarce new drug.'],
      sure='A strategy that is certain to cure AMT patients',
      lot='A strategy with a PROB chance of curing AMT patients, and no cures otherwise'),
    S(domain='employment', type='money', role='consultant', what=('payment term', 'payment terms'),
      ctx=['%(S)s %(be)s negotiating a consulting contract and can choose between two payment terms.',
           '%(S)s %(have)s been hired for a short advisory project and %(be)s offered two ways to be paid.'],
      sure='A fixed payment of $AMT, guaranteed',
      lot='A contingent payment with a PROB chance of paying $AMT, and nothing otherwise'),
    S(domain='gambling', type='money', role='shopper', what=('panel', 'panels'),
      ctx=['%(S)s %(have)s a promotional scratch card with two panels and may scratch only one.',
           'After a purchase, %(s1)s receives a scratch card and may scratch one of its two panels.'],
      sure='A panel that reveals a prize of $AMT for certain',
      lot='A panel with a PROB chance of revealing a prize of $AMT, and nothing otherwise'),
]

ASK = [  # (part 0, part 1): different wording, same meaning; neither mentions the other part
    ('Which %(what)s %(Q)s choose?', 'Which of the following %(whats)s %(Q)s choose?'),
    ('Which of the two %(whats)s %(Q)s take?', 'Which %(what)s %(Q)s pick?'),
]


def render(sc: dict, ci: int, tok: dict, rng) -> list[Item]:
    """two_part: part 0 A = X sure vs B = (Y, p); part 1 C = (X, k) vs D = (Y, k*p). Sure side first.
    stated_prior (x2): states the part-0 choice (A or B), asks the part-1 decision; EU: A -> C, B -> D."""
    tok = dict(tok, what=sc['what'][0], whats=sc['what'][1])
    ask0, ask1 = ASK[rng.randrange(len(ASK))]
    context = fill_context(sc['ctx'][ci], tok)
    p0 = Part(f'{context} {ask0 % tok}', [opt(sc['sure'], None, 'x_amount'), opt(sc['lot'], 'p_pct', 'y_amount')])
    p1 = Part(f'{context} {ask1 % tok}', [opt(sc['lot'], 'k_pct', 'x_amount'), opt(sc['lot'], 'kp_pct', 'y_amount')])
    stated = stated_prior_items(p0, now_context(sc['ctx'][ci], tok, sc['role']), ask1 % tok, p1, tok, tok['whats'])
    return [Item('two_part', '', [p0, p1])] + stated


def main():
    build_dataset('certainty_effect', HERE, SCENARIOS, EXAMPLE_SCENARIOS, render, SEED, DIFFICULTY)


if __name__ == '__main__':
    main()

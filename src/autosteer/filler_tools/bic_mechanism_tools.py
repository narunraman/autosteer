"""bic_mechanism (STEER 4.3.b, Bayesian Incentive Compatibility): hand-written templates.

Design of 6 October 2026. Independent private values with a stated distribution (uniform from $0, or for the
expected-externality mechanism net benefits uniform on a symmetric interval), risk-neutral participants who know the
distribution. Mechanisms are described in words, never named; the type is the mechanism. Yes/No: is truthful
reporting (bidding one's value; for the rising-price auction, staying in until the price reaches one's value) a
Bayes-Nash equilibrium?

    type               BIC  why
    second_price       Yes  dominant strategy, hence a Bayes-Nash equilibrium
    english            Yes  (announced drop-outs, private values) staying in to one's value is dominant
    posted_report      Yes  report a value; among reports at or above a fixed price one buyer is drawn and pays the
                            price: truthful reporting is weakly dominant
    agv                Yes  expected-externality (AGV / d'Aspremont-Gerard-Varet) public project: each resident is paid
                            the expected net benefit of the others given their own report, financed by the others in
                            equal shares; truthful reporting maximises the resident's expected total surplus when the
                            others are truthful (Bayes-Nash, not dominant)
    scaled_payment     Yes  when the winner pays (n-1)/n of their own report (the direct version of the first-price
                            equilibrium): with others truthful, U(r) = (v - k r)(r/V)^(n-1) is maximised at
                            r = (n-1) v / (n k) = v. No when k is another fraction (half the questions).
    first_price        No   with others bidding their values, a lower bid still wins often and pays less
    third_price        No   (three or more bidders) bidding above one's value pays off when the third bid is low
    pay_your_bid       No   public project paid by pledges; a truthful pledger who is decisive gets nothing, pledging
                            less keeps a positive chance of the project at a lower payment
    proportional_share No   cost shared in proportion to reports: lowering a report lowers the share, and at the
                            truthful report the loss from the project failing is zero, so shading is profitable
    bne_bid            (second type) first-price auction, n bidders with values uniform on [0, V]: the symmetric
                       equilibrium bid of a bidder with value v is (n - 1) v / n; four whole-dollar options
                       (distractors: v itself, v / n, v / 2, n v / (n + 1), (n - 1) V / n).
Yes and No are equally common (four Yes types, four No types and scaled_payment half and half; auction and project
types have the same number of templates). Perspectives: anon, first_person, second_person, ai_assistant (advising the
seller or the organiser); bne_bid: second_person, first_person, man, woman (the bidder).
"""
from __future__ import annotations

from math import lcm

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils
from .utilities.story_utils import MEN, WOMEN

function_registry = FillRegister()

ELEMENT = 'bic_mechanism'
AUCTIONS = [  # seller, item, domain
    ('an art gallery', 'a painting', 'art'),
    ('a city', 'a five-year lease on a market stall', 'government'),
    ('a farmer', 'a used tractor', 'agriculture'),
    ('a charity', 'a signed guitar', 'charity'),
]
PROJECTS = [  # group, member, project, domain
    ('the residents of Maple Street', 'resident', 'a shared solar battery', 'community'),
    ('the employees of a design studio', 'employee', 'a new espresso machine', 'workplace'),
    ('the parents at Oakridge Elementary', 'parent', 'a shaded play area', 'school'),
    ('the members of a rowing club', 'member', 'a new boat shed', 'sports'),
]
AUCTION_TYPES = ('second_price', 'english', 'posted_report', 'scaled_payment', 'first_price', 'third_price')
PROJECT_TYPES = ('agv', 'pay_your_bid', 'proportional_share')
BIC = {'second_price': True, 'english': True, 'posted_report': True, 'agv': True, 'first_price': False,
       'third_price': False, 'pay_your_bid': False, 'proportional_share': False}
LEVEL = {'second_price': 1, 'first_price': 1, 'english': 1, 'posted_report': 1, 'third_price': 2,
         'scaled_payment': 2, 'agv': 3, 'pay_your_bid': 3, 'proportional_share': 3, 'bne_bid': 2}
FRACTIONS = {(1, 2): 'half', (2, 3): 'two-thirds', (3, 4): 'three-quarters', (4, 5): 'four-fifths'}
NUM_WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five'}

MECH_TEXT = {
    'second_price': "Each bidder submits a sealed bid; the highest bidder wins and pays the second-highest bid.",
    'first_price': "Each bidder submits a sealed bid; the highest bidder wins and pays their own bid.",
    'third_price': "Each bidder submits a sealed bid; the highest bidder wins and pays the third-highest bid.",
    'english': ("The price starts at $0 and rises continuously. Each bidder may drop out at any moment, and every "
                "drop-out is announced to all. When only one bidder is left, that bidder wins and pays the price at "
                "which the last other bidder dropped out."),
    'posted_report': ("Each bidder reports a value in a sealed envelope. Among the bidders who report at least "
                      "{price}, one is drawn at random; that bidder gets the {short} and pays {price}. If no one "
                      "reports at least {price}, the {short} is not sold."),
    'scaled_payment': ("Each bidder reports a value in a sealed envelope. The bidder with the highest report gets the "
                       "{short} and pays {fraction} of their own report."),
    'agv': ("Each {member} reports a net benefit. The project goes ahead if the reported net benefits add up to more "
            "than zero. In addition, each {member} receives a payment equal to the expected total net benefit that the "
            "other {member}s get from the decision, calculated from that {member}'s own report and the distribution "
            "of the others' net benefits (not from their reports); this payment is collected from the other "
            "{member}s in equal shares."),
    'pay_your_bid': ("Each {member} pledges an amount. If the pledges add up to at least the cost of {cost}, the "
                     "project goes ahead and each {member} pays exactly the amount they pledged (anything above the "
                     "cost goes to a reserve fund); otherwise nothing is built and nobody pays."),
    'proportional_share': ("Each {member} reports a value. If the reports add up to at least the cost of {cost}, the "
                           "project goes ahead and each {member} pays a share of the cost proportional to their "
                           "report; otherwise nothing is built and nobody pays."),
}
TRUTH_Q = {
    'english': ("Is it a Bayes-Nash equilibrium for every bidder to stay in until the price reaches their own value?"),
    'auction': "Is it a Bayes-Nash equilibrium for every bidder to bid (or report) exactly their own value?",
    'project': "Is it a Bayes-Nash equilibrium for every {member} to {verb} exactly their own {what}?",
}


def _advisor(persp: str, who: str, activity: str) -> str:
    """'An art gallery is selling a painting.' / 'You are advising the residents of ..., who are deciding ...'."""
    plural = who.startswith('the ')
    be = 'are' if plural else 'is'
    rel = 'who' if plural or who == 'a farmer' else 'which'
    if persp == 'anon':
        return f'{who[0].upper() + who[1:]} {be} {activity}.'
    lead = {'first_person': 'I am advising', 'second_person': 'You are advising',
            'ai_assistant': 'You are an AI assistant advising'}[persp]
    return f'{lead} {who}, {rel} {be} {activity}.'


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for persp in ('anon', 'first_person', 'second_person', 'ai_assistant'):
        for seller, item, dom in AUCTIONS:
            short = item.split(' on ')[0].replace('a five-year lease', 'lease').split()[-1]
            head = (_advisor(persp, seller, f'selling {item}') + " There are {n_word} bidders. Each bidder's value "
                    "for the " + short + " is private; the values are independent and uniformly distributed between "
                    "$0 and {vmax}, the bidders are risk-neutral, and all of this is common knowledge. The selling "
                    "rule: ")
            for t in AUCTION_TYPES:
                body = MECH_TEXT[t].replace('{short}', short)
                q = TRUTH_Q['english' if t == 'english' else 'auction']
                out.append(ct.CodeTemplate(parts=(head + body + ' ' + q,), type=t, domain=dom, tags=persp,
                                           difficulty_level=LEVEL[t], spec={'type': t}))
        for group, member, project, dom in PROJECTS:
            for t in PROJECT_TYPES:
                what = 'net benefit' if t == 'agv' else 'value'
                if t == 'agv':
                    dist = (f"Each {member}'s net benefit from {project} (their benefit minus their share of its "
                            f"cost, which can be negative) is private; the net benefits are independent and uniformly "
                            f"distributed between -{{vmax}} and {{vmax}}")
                else:
                    dist = (f"Each {member}'s value for {project} is private; the values are independent and "
                            f"uniformly distributed between $0 and {{vmax}}")
                head = (_advisor(persp, group, f'deciding whether to fund {project}') + " There are {n_word} "
                        f"{member}s. {dist}, everyone is risk-neutral, and all of this is common knowledge. The "
                        "decision rule: ")
                body = MECH_TEXT[t].replace('{member}', member)
                q = TRUTH_Q['project'].replace('{member}', member).replace('{what}', what).replace(
                    '{verb}', 'pledge' if t == 'pay_your_bid' else 'report')
                out.append(ct.CodeTemplate(parts=(head + body + ' ' + q,), type=t, domain=dom, tags=persp,
                                           difficulty_level=LEVEL[t], spec={'type': t}))
    for persp in ('second_person', 'first_person', 'man', 'woman'):
        for k, (seller, item, dom) in enumerate(AUCTIONS):
            short = item.split(' on ')[0].replace('a five-year lease', 'lease').split()[-1]
            if persp == 'second_person':
                subj, val, q = 'You are', 'You value', 'what do you bid'
            elif persp == 'first_person':
                subj, val, q = 'I am', 'I value', 'what should I bid'
            else:
                name = (MEN if persp == 'man' else WOMEN)[k % 7]
                subj, val, q = f'{name} is', f'{name} values', f'what does {name} bid'
            text = (f"{subj} one of {{n_word}} bidders in a sealed-bid auction for {item} run by {seller}: the highest "
                    f"bidder wins and pays their own bid. Each bidder's value is private; the values are independent "
                    f"and uniformly distributed between $0 and {{vmax}}, the bidders are risk-neutral, and all of this "
                    f"is common knowledge. {val} the {short} at {{value}}. In the symmetric Bayes-Nash equilibrium, "
                    f"{q}?")
            out.append(ct.CodeTemplate(parts=(text,), type='bne_bid', domain=dom, tags=persp, difficulty_level=2,
                                       spec={'type': 'bne_bid'}))
    return out


def money(x: int) -> str:
    return f'${x:,}'


@function_registry.register(ELEMENT)
def fill_bic_mechanism(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    t = spec['type']
    r = filler_utils.rng
    if t == 'third_price':
        n = int(r.integers(3, 6))
    else:
        n = int(r.integers(2, 6))
    out = {'n_word': NUM_WORDS[n]}
    if t == 'bne_bid':
        step = lcm(n, n + 1, 2) * 5
        v = step * int(r.integers(max(1, 40 // step + 1), 4000 // step + 1))
        vmax = 100 * int(r.integers(v // 100 + 1, v // 100 + 40))
        key = v * (n - 1) // n
        pool = [v, v // n, v // 2, v * n // (n + 1), vmax * (n - 1) // n]
        opts = [key]
        for i in r.permutation(len(pool)):
            x = pool[int(i)]
            if x not in opts and x > 0:
                opts.append(x)
            if len(opts) == 4:
                break
        out.update({'value': money(v), 'vmax': money(vmax), 'options_list': [[money(x) for x in opts]],
                    'correct_answer': [[1, 0, 0, 0]]})
        return out
    vmax = 10 * int(r.integers(5, 101))
    out['vmax'] = money(vmax)
    if t == 'scaled_payment':
        right = (n - 1, n)
        yes = bool(r.random() < 0.5)
        if yes:
            frac = right
        else:
            others = [f for f in FRACTIONS if f[0] * n != f[1] * (n - 1)]
            frac = others[int(r.integers(len(others)))]
        out['fraction'] = FRACTIONS[frac]
    else:
        yes = BIC[t]
    out['price'] = money(10 * int(r.integers(1, vmax // 10)))
    out['cost'] = money(10 * int(r.integers(max(1, n * vmax // 40), max(2, n * vmax // 15))))
    out['options_list'] = [['Yes', 'No']]
    out['correct_answer'] = [[int(yes), int(not yes)]]
    return out


ct.register(ELEMENT, _templates())

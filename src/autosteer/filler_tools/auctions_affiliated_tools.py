"""auctions_affiliated (STEER 4.4.c, Optimal Auction for Bidders with Affiliated Values): hand-written templates.

Design of 6 October 2026. The environment is never named: it has to be read from the story.
    affiliated   the item is worth about the same to every bidder (oil under a tract, laptops to be resold, a wine
                 lot for resale, timber) and each bidder has its own noisy estimate; risk-neutral bidders.
                 Expected revenue (linkage principle, symmetric bidders, three or more): English > second-price >
                 first-price = Dutch.
    ipv_neutral  the value is the bidder's own enjoyment, the item cannot be resold and values are unrelated;
                 risk-neutral bidders. Revenue equivalence: all four formats equal.
    ipv_averse   private values as above, and every bidder would rather have a sure $50 than an even chance of $100
                 (risk aversion, not named): first-price = Dutch > second-price = English.
Each story has an explicit and a subtle wording of its cues (the subtle one gives the facts without the gloss: e.g.
"each company's geologists studied their own seismic survey" instead of "the oil is worth the same to whoever wins,
and each survey is a noisy estimate"). The formats are described, never named, and labelled Format 1 to 4 in a
random order: a rising price clock with announced drop-outs (English), sealed bids paying the second-highest bid,
sealed bids paying one's own bid, a falling price clock stopped by the first bidder (Dutch).

Types:
    ranking3 / ranking4: rank three or all four formats by the seller's expected revenue, ties shown with "=";
        options are orderings. Distractors: the orderings of the two other environments, the reversed ordering, the
        affiliated ordering with the first-price / Dutch tie broken or English tied with second-price, all restricted
        to the listed formats and distinct.
    info_policy: should the seller commit to publishing an independent expert appraisal before the auction? Yes
        (raises expected revenue) under affiliation; no difference under independent private values.
Difficulty: ranking3 with explicit cues 1, ranking3 subtle or ranking4 explicit 2, ranking4 subtle 3; info_policy
explicit 1, subtle 2. Half the stories are affiliated, so info_policy is Yes half the time.
Perspectives: anon, first_person, second_person, ai_assistant (advising the seller).
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

ELEMENT = 'auctions_affiliated'
NEUTRAL = 'They care only about their expected profit.'
STORIES = [  # env, domain, thing, explicit cues, subtle cues ({n} = number word)
    ('affiliated', 'energy', 'tract',
     "A government is auctioning the drilling rights to an offshore oil tract to {n} oil companies. The oil under the "
     "tract is worth the same to whichever company wins, but nobody knows how much there is: each company has run its "
     "own seismic survey, which gives it a noisy estimate. " + NEUTRAL,
     "A government is auctioning the drilling rights to an offshore oil tract to {n} oil companies. Each company's "
     "geologists have studied their own seismic survey of the tract, and their estimates differ; the winner will pump "
     "the oil and sell it at the world price. " + NEUTRAL),
    ('affiliated', 'retail', 'pallet',
     "A retailer is auctioning a pallet of returned laptops to {n} resellers. Every reseller will sell the laptops "
     "on at the same market prices, so the pallet is worth about the same to each of them, but nobody knows how many "
     "laptops work: each reseller has tested a different handful, which gives it an imperfect estimate. " + NEUTRAL,
     "A retailer is auctioning a pallet of returned laptops to {n} resellers, who will refurbish them and sell them "
     "on. Before bidding, each reseller was allowed to test a different handful of the laptops. " + NEUTRAL),
    ('affiliated', 'collectibles', 'wine lot',
     "An estate is auctioning a lot of vintage wine to {n} wine merchants, who will all resell it in the same "
     "market, so the lot is worth about the same to each of them. Nobody knows its quality for sure: each merchant "
     "has tasted different bottles, which gives it an imperfect estimate. " + NEUTRAL,
     "An estate is auctioning a lot of vintage wine to {n} wine merchants, each of whom would resell it at the "
     "going trade price for such wine. "
     "Each merchant has tasted a few different bottles from the lot. " + NEUTRAL),
    ('affiliated', 'forestry', 'tract',
     "A province is auctioning the logging rights to a forest tract to {n} sawmills. The timber will fetch the same "
     "price whichever mill cuts it, but nobody knows how much usable timber there is: each mill's forester has "
     "surveyed a different part of the tract and has a noisy estimate. " + NEUTRAL,
     "A province is auctioning the logging rights to a forest tract to {n} sawmills, which sell their lumber at "
     "market prices. Each mill's forester has walked a different part of the tract to count usable trees. "
     + NEUTRAL),
    ('ipv_neutral', 'charity', 'dinner',
     "A charity is auctioning a private dinner with a famous chef to {n} donors. Each donor's value comes only from "
     "their own enjoyment of the evening: the dinner cannot be resold, and one donor's value says nothing about "
     "another's. " + NEUTRAL,
     "A charity is auctioning a private dinner with a famous chef to {n} donors. The dinner is booked in the winner's "
     "name and cannot be passed on, and the donors' tastes in food are their own and unrelated to one another's. "
     + NEUTRAL),
    ('ipv_neutral', 'entertainment', 'role',
     "A theatre is auctioning a walk-on role in its next play to {n} fans. Each fan's value comes only from their "
     "own enjoyment of being on stage: the role cannot be transferred, and the fans' values are unrelated to one "
     "another. " + NEUTRAL,
     "A theatre is auctioning a walk-on role in its next play to {n} fans. The role goes to the winner in person, "
     "and how much each fan would enjoy it is a private matter that has nothing to do with the other fans. "
     + NEUTRAL),
    ('ipv_averse', 'sports', 'ball',
     "A football club is auctioning a signed match ball to {n} fans. Each fan's value comes only from the pleasure "
     "of owning it: the ball will not be resold, and the fans' values are unrelated to one another. Each fan dislikes "
     "risk: they would always rather have a gamble's average amount for sure than the gamble itself (for example, a "
     "sure $50 rather than a fifty-fifty chance of $100 or nothing).",
     "A football club is auctioning a signed match ball to {n} fans who want it for their own shelves; how much each "
     "fan cares about it is personal and unrelated to the others. The fans are cautious: each of them strictly "
     "prefers a gamble's average amount for sure to any risky gamble (a sure $50 over a coin flip for $100, say)."),
    ('ipv_averse', 'travel', 'cabin',
     "A ferry company is auctioning the last private cabin on a holiday crossing to {n} travellers. Each traveller's "
     "value comes only from their own comfort on the trip: the booking cannot be resold, and the travellers' values "
     "are unrelated to one another. Each traveller dislikes risk: they would always rather have a gamble's average "
     "amount for sure than the gamble itself (for example, a sure $50 rather than a fifty-fifty chance of $100 or "
     "nothing).",
     "A ferry company is auctioning the last private cabin on a holiday crossing to {n} travellers. The booking is "
     "in the winner's name, and how much the cabin is worth to each traveller depends only on their own plans. Every "
     "traveller strictly prefers a gamble's average amount paid for sure to any risky gamble (a sure $50 to a coin "
     "flip for $100, say)."),
]
FORMATS = {
    'E': ("a price clock starts low and rises; bidders drop out one by one, every drop-out is announced to all, and "
          "the last bidder left wins and pays the price at which the previous bidder dropped out"),
    'S': "each bidder submits a sealed bid; the highest bidder wins and pays the second-highest bid",
    'F': "each bidder submits a sealed bid; the highest bidder wins and pays their own bid",
    'D': "a price clock starts high and falls; the first bidder to stop it wins and pays the price shown",
}
# revenue rank (higher is more revenue) by environment
REVENUE = {
    'affiliated': {'E': 3, 'S': 2, 'F': 1, 'D': 1},
    'ipv_neutral': {'E': 1, 'S': 1, 'F': 1, 'D': 1},
    'ipv_averse': {'E': 1, 'S': 1, 'F': 2, 'D': 2},
}
DISTRACTOR_SCORES = [
    {'E': 1, 'S': 2, 'F': 3, 'D': 3},  # the affiliated order reversed
    {'E': 3, 'S': 3, 'F': 1, 'D': 1},  # English tied with second-price
    {'E': 4, 'S': 3, 'F': 2, 'D': 1},  # the first-price / Dutch tie broken
    {'E': 4, 'S': 3, 'F': 1, 'D': 2},
    {'E': 2, 'S': 3, 'F': 1, 'D': 1},  # second-price above English
    {'E': 2, 'S': 2, 'F': 1, 'D': 1},
]
INFO_OPTIONS = ['Yes: publishing it raises the expected revenue', 'No: publishing it lowers the expected revenue',
                'It makes no difference to the expected revenue']
NUM = {3: 'three', 4: 'four', 5: 'five', 6: 'six'}
# The standard assumptions under which the three rankings hold (review, October 2026: symmetry, equilibrium
# bidding and the reserve were not stated). The same sentence in every story, so it does not tell the environment.
# The information environment, stated in every story (re-review, October 2026: the stories did not establish
# affiliation or what the appraisal is informative about). It is the premise; the question is what it implies.
ENVIRONMENT = {
    'affiliated': ("Each bidder's estimate is a noisy signal of the same unknown value. The bidders' signals and the "
                   "item's value are strictly affiliated: a higher value of any one makes higher values of the others "
                   "more likely."),
    'ipv_neutral': "Each bidder already knows their own value.",
    'ipv_averse': "Each bidder already knows their own value.",
}
APPRAISAL = {
    'affiliated': ("The appraisal is another noisy signal of the same unknown value, strictly affiliated with the "
                   "value and the bidders' signals in the same way."),
    'ipv_neutral': "The appraisal says nothing about how much any bidder values it.",
    'ipv_averse': "The appraisal says nothing about how much any bidder values it.",
}
SYMMETRIC = ("The bidders are alike: each one's information about the item is drawn in the same way, all of them bid as "
             "in the symmetric equilibrium of each format, and there is no reserve price.")


def _lead(persp: str) -> str:
    return {'anon': '', 'first_person': 'I am advising the seller in this auction. ',
            'second_person': 'You are advising the seller in this auction. ',
            'ai_assistant': 'You are an AI assistant advising the seller in this auction. '}[persp]


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for s, (env, dom, thing, explicit, subtle) in enumerate(STORIES):
        for cue, story in (('explicit', explicit), ('subtle', subtle)):
            for persp in ('anon', 'first_person', 'second_person', 'ai_assistant'):
                head = _lead(persp) + story.replace('{n}', '{n_word}') + ' ' + ENVIRONMENT[env] + ' ' + SYMMETRIC + ' '
                for k in (3, 4):
                    text = (head + f"The seller is considering {NUM[k]} formats:\n{{formats}}\nRank these formats by "
                            "the seller's expected revenue, highest first (\"=\" means equal expected revenue).")
                    level = {('explicit', 3): 1, ('subtle', 3): 2, ('explicit', 4): 2, ('subtle', 4): 3}[(cue, k)]
                    out.append(ct.CodeTemplate(parts=(text,), type=f'ranking{k}', domain=dom, tags=f'{persp};{cue}',
                                               difficulty_level=level,
                                               spec={'story': s, 'env': env, 'type': 'ranking', 'k': k}))
                text = (head + "The seller will use this format: {format}. The seller also holds an independent "
                        f"expert's appraisal of the {thing}. {APPRAISAL[env]} Should the seller commit to publishing the appraisal "
                        "to all bidders before the auction?")
                out.append(ct.CodeTemplate(parts=(text,), type='info_policy', domain=dom, tags=f'{persp};{cue}',
                                           difficulty_level=1 if cue == 'explicit' else 2,
                                           spec={'story': s, 'env': env, 'type': 'info_policy'}))
    return out


def ordering_text(score: dict, label: dict) -> str:
    """'Format 2 > Format 1 = Format 4' from scores (higher first); ties sorted by label."""
    tiers = sorted(set(score.values()), reverse=True)
    return ' > '.join(' = '.join(sorted(f'Format {label[f]}' for f in score if score[f] == v)) for v in tiers)


@function_registry.register(ELEMENT)
def fill_auctions_affiliated(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    r = filler_utils.rng
    env = spec['env']
    out = {'n_word': NUM[int(r.integers(3, 7))]}
    if spec['type'] == 'info_policy':
        fmt = 'ESF'[int(r.integers(3))]
        out['format'] = FORMATS[fmt]
        key = INFO_OPTIONS[0] if env == 'affiliated' else INFO_OPTIONS[2]
        out['options_list'] = [INFO_OPTIONS]
        out['correct_answer'] = [[int(o == key) for o in INFO_OPTIONS]]
        return out
    k = spec['k']
    while True:
        if k == 4:
            fmts = ['E', 'S', 'F', 'D']
        else:
            fmts = [['E', 'S', 'F'], ['E', 'S', 'D'], ['E', 'F', 'D'], ['S', 'F', 'D']][int(r.integers(4))]
        order = [fmts[int(i)] for i in r.permutation(k)]
        label = {f: i + 1 for i, f in enumerate(order)}
        key = ordering_text({f: REVENUE[env][f] for f in fmts}, label)
        pool = [ordering_text({f: sc[f] for f in fmts}, label)
                for sc in [REVENUE[e] for e in REVENUE if e != env] + DISTRACTOR_SCORES]
        distinct = []
        for i in r.permutation(len(pool)):
            o = pool[int(i)]
            if o != key and o not in distinct:
                distinct.append(o)
        if len(distinct) >= 3:
            break
    out['formats'] = '\n'.join(f'Format {i + 1}: {FORMATS[f][0].upper() + FORMATS[f][1:]}.'
                               for i, f in enumerate(order))
    out['options_list'] = [[key] + distinct[:3]]
    out['correct_answer'] = [[1, 0, 0, 0]]
    return out


ct.register(ELEMENT, _templates())

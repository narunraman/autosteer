"""The conventions and wordings the stored templates carry, for the tests that check them.

They were applied at load time by the sampler (template_fixes, consumer_template_fixes, behav_fixes) until
October 2026, when the corrected templates were stored in data/ (scripts/migrations/fold_template_fixes.py)."""
from typing import Optional
import re


def _names_x_and_y(text: str) -> bool:
    """The text says which good is x and which is y ("pizza (x)", "'x' represents pizza")."""
    def named(v):
        return bool(re.search(rf'\({v}\)', text) or
                    re.search(rf"'?\b{v}\b'? (?:is|represents|denotes|stands for|refers to)\b", text))
    return named('x') and named('y')


_Y_NAMED = [
    # "... and y denotes the consumption of electronics."
    re.compile(r'\by (?:denotes|represents|signifies|stands for|is|refers to|indicates|measures|captures)(?: the)? ([^.,;:()]+?)\s*(?:[.,;:]|$)'),
    # "... online lectures (y)"
    re.compile(r'(?:and|,)\s+([\w\- ]+?) \(y\)'),
]


_BETWEEN = re.compile(r'between (?:two [\w\- ]+?: )?([\w\- ]+?) and ([\w\- ]+?)[.,?]')


def _strip_article(s: str) -> str:
    return re.sub(r'^(the|a|an)\s+', '', s.strip(), flags=re.I)


def y_good_name(text: str) -> Optional[str]:
    """Best-effort name of the good called y in a deriving_demand template."""
    for pat in _Y_NAMED:
        m = pat.search(text)
        if m:
            return _strip_article(m.group(1))
    m = _BETWEEN.search(text)
    if m:
        return _strip_article(m.group(2))
    return None


CONSUMER_SURPLUS_CONVENTION = ('The demand function gives the price P buyers are willing to pay as a function of the '
                               'quantity Q (an inverse demand curve), and the good is treated as perfectly divisible.')


DWL_CONVENTION = ('Here both curves give the price P as a function of the quantity Q, the supply curve is the '
                  "seller's marginal cost curve, and the seller charges one price for every unit. The deadweight loss "
                  'is the total surplus lost relative to the competitive quantity, where demand equals marginal cost.')


DWL_PRICE_EXACT = 'Take the stated monopoly price as exact.'


EFFICIENT_SURPLUS_CONVENTION = 'Here both functions give the price P as a function of the quantity Q.'


EFFICIENT_SURPLUS_PRICE_ROUNDED = ('The stated equilibrium price is rounded to two decimal places; use the exact '
                                   'intersection of the two curves.')


INTERTEMPORAL_CONVENTION = (
    'Assume the initial wealth and the first income are available at the start of the first year, each income '
    'arrives and each year\'s consumption takes place at the start of that year, savings earn the interest rate '
    'between years, and nothing is left at the end. Lifetime utility is the discounted sum over the years of '
    'u(c) = c^(1-g)/(1-g), where g is the coefficient of relative risk aversion and each later year is weighted by '
    'one more power of the discount factor. Each option lists consumption in each year, in order.')


STATE_BASED_CONVENTION = (
    'The dollar amounts spent on the two goods must be fixed now, before it is known which state occurs; in either '
    'state those dollars buy the goods at that state\'s prices, and the goods are divisible. In the utility function, '
    'x is the quantity of the first good and y the quantity of the second, in the order the prices are listed, and '
    'each option gives (dollars spent on the first good, dollars spent on the second good).')


# Welfare theorems: the stories only say someone is "satisfied with the market prices" or that "no trade could make
# both better off"; the premise each theorem needs is now stated explicitly before the question.
WT1_PREMISE = ('Assume the goods are divisible and the market reaches a competitive equilibrium: at the market '
               'prices every participant ends up '
               'with the bundle they like best among those they can afford, every market clears, more of a good is '
               'always better, and nobody is affected by what others hold or consume (there are no externalities).')


WT2_PREMISE = ('Assume the goods are divisible, everyone\'s preferences are continuous, convex and strictly '
               'increasing in each good, and the final allocation is Pareto efficient: no reallocation of the goods '
               'could make someone better off without making someone else worse off.')


WT2_WITH = ('According to the second welfare theorem, is there a lump-sum redistribution of the initial holdings and '
            'a set of competitive equilibrium prices under which this same final allocation is a competitive '
            'equilibrium?')


WT2_WITHOUT = ('Without any redistribution of the initial holdings, is it guaranteed that this same final allocation '
               'is a competitive equilibrium at some set of prices?')


# goods whose trade affects third parties (the first welfare theorem assumes no externalities)
_WT_EXTERNALITY = re.compile(r'pollut|emission|carbon|water (?:[\w-]+ )?rights?|cap[- ]and[- ]trade|greenhouse', re.I)


# stories without goods to trade (task assignments, tutoring, schedules): no exchange economy
_WT_NO_GOODS = re.compile(r'\b(tasks?|tutor\w*|chores?|sessions?|lessons?|shifts?|schedul\w*|duties|'
                          r'responsibilit\w*|understanding|skills?|hours|time slots?|roles?)\b', re.I)


WT1_FURTHER = ('After the market reaches this equilibrium, is there any further trade that would make some '
               'participant better off without making anyone worse off?')


# complement_rule: question wordings whose event is not the complement of the given one(s) (a game or a match can
# be drawn, a flight or a delivery can be early, hand preference and eye colour have more than the named values, an
# adoption need not be of exactly one animal, home runs are often fly balls); the question is reworded so the asked
# event is the complement, or the setting so the named events are exhaustive and exclusive.
COMPLEMENT_REWORDS = [
    ('What is the probability that the flight will be delayed?', 'What is the probability that the flight will not be on time?'),
    ("What's the probability that they are right-handed?", "What's the probability that they are not left-handed?"),
    ("What's the probability your food delivery will be late?", "What's the probability your food delivery will not be punctual?"),
    ('What is the probability that your flight will be late?', 'What is the probability that your flight will not be on time?'),
    ("What's the probability that you will win the game?", "What's the probability that the champion will not win the game?"),
    ('What is the probability you will lose?', 'What is the probability you will not win?'),
    ('What is the probability of you losing?', 'What is the probability of you not winning?'),
    ('What is the probability that it will be late?', 'What is the probability that it will not arrive on time?'),
    ('What is the probability to inherit green eyes?', 'What is the probability to inherit an eye colour other than brown or blue?'),
    ('What is the probability a student is a major in another field?',
     'Each student has exactly one major. What is the probability a student is a major in another field?'),
    ('In an animal shelter, there is a {PA} probability of adopting a dog and a {PB} probability of adopting a cat. '
     'What is the probability of adopting another type of animal?',
     'Each adoption at an animal shelter is of exactly one animal. There is a {PA} probability that an adopted animal '
     'is a dog and a {PB} probability that it is a cat. What is the probability that it is another type of animal?'),
    ('You hit a baseball with a {PA} chance of hitting a home run and a {PB} chance of hitting a ground ball. What is '
     'the probability of hitting a fly ball?',
     'Every ball you hit into play is exactly one of a ground ball, a line drive or a fly ball. There is a {PA} chance '
     'of hitting a ground ball and a {PB} chance of hitting a line drive. What is the probability of hitting a fly '
     'ball?'),
]


def exact_copies(q, o) -> set:
    """question_ids of the templates that repeat an earlier one exactly: every part's text, the template's options and
    its metadata. The data has none (they were deleted from it), but the corrections can make some:
    welfare_theorem_1 and _2 give every template of a type one fixed question, so templates that differed only in
    their question are the same in that type (and still differ in the others, so they stay in the data)."""
    if q is None or len(q) == 0:
        return set()
    if 'question_id' not in q.columns:
        q = q.reset_index()
    if o is not None and len(o) and 'question_id' not in o.columns:
        o = o.reset_index()
    qs = q.sort_values(['question_id', 'sub_id'], kind='stable')
    meta = ['domain', 'type', 'difficulty_level', 'tags']
    g = qs.groupby('question_id', sort=True)
    key = g['question_text'].agg(tuple).to_frame('text').join(g[meta].first().astype(str))
    if o is not None and len(o):
        os_ = o.sort_values(['question_id', 'sub_id', 'option_id'], kind='stable')
        opt = (os_['sub_id'].astype(str) + '\x1f' + os_['option_text'].astype(str) + '\x1f'
               + os_['correct_answer'].astype(str)).groupby(os_['question_id']).agg(tuple)
        key = key.join(opt.rename('options'))
        key['options'] = key['options'].apply(lambda v: v if isinstance(v, tuple) else ())
    return set(key.index[key.apply(tuple, axis=1).duplicated(keep='first')].tolist())


# ---------------------------------------------------------------------------------------------------------------
# deriving_demand
# ---------------------------------------------------------------------------------------------------------------
_VERB = r'(?:is|are|represents?|denotes?|stands? for|refers? to|signif(?:y|ies)|indicates?|measures?|captures?|for)'


def names_good(text: str, v: str) -> bool:
    """The text says what the variable v (x or y) stands for."""
    return bool(re.search(rf"\({v}\)|'{v}'|\b{v} {_VERB}\b|(?:denoted|represented|labell?ed|indicated) (?:by|as) "
                          rf"'?{v}\b|\b{v} (?:units|hours|amount)|\bby '?{v}\b", text))


DERIVING_DEMAND_PRICES = 'Here p_x and p_y are the prices of one unit of x and of y.'


# ---------------------------------------------------------------------------------------------------------------
# marginal_utility, mrs_utility
# ---------------------------------------------------------------------------------------------------------------
MARGINAL_UTILITY_FORMAT = ('Answer with the pair (MU_x, MU_y), the partial derivatives of U with respect to x and '
                           'to y.')


MRS_FORMAT = ('Answer with the ordered pair (MU_x/MU_y, MU_y/MU_x), where MU_x and MU_y are the marginal utilities '
              'of x and y (x, y > 0).')


# ---------------------------------------------------------------------------------------------------------------
# aggregation_of_consumer_demand, aggregation_of_offer_curve
# ---------------------------------------------------------------------------------------------------------------
AGG_DEMAND_CONVENTION = ('Each demand function gives the quantity demanded by a single member of its group at price '
                         'P; the answer is the total over both groups, rounded to two decimal places.')


AGG_OFFER_CONVENTION = ('Each offer curve gives the quantity offered by a single member of its group at price P; '
                        'the answer is the total over both groups (numbers rounded to two decimal places).')


# ---------------------------------------------------------------------------------------------------------------
# law_of_demand
# ---------------------------------------------------------------------------------------------------------------
LAW_OF_DEMAND_CONVENTION = ('Assume nothing else that affects demand has changed (incomes, tastes, the prices of '
                            'other goods), so the change is a movement along an unchanged, downward-sloping demand '
                            'curve; exactly one of the values below is consistent with this.')


# ---------------------------------------------------------------------------------------------------------------
# deriving_hicksian_demand
# ---------------------------------------------------------------------------------------------------------------
HICKSIAN_ROUNDING = 'Numbers in the options are rounded to two decimal places.'


# ----------------------------------------------------------------------
# gamblers_fallacy
# ----------------------------------------------------------------------
_OTHER = 'asks whether a different outcome is more likely than the streak outcome, not whether it became more likely'


_DEPENDENT = 'cards dealt from a shoe or deck: successive hands are not independent'


_NO_CHANGE = 'does not ask whether the probability changed (no comparison or no horizon)'


GAMBLERS_FALLACY_DROPS: dict[int, str] = {
    **{i: _OTHER for i in (32, 44, 65, 71, 93, 95, 116, 121, 122, 123, 144)},
    145: 'a total of seven is more likely than six with two dice, so "Yes" is defensible',
    **{i: _DEPENDENT for i in (10, 17, 23, 24, 45, 64, 80, 86, 129, 135, 136)},
    40: 'poker hands won depend on skill, not independent fair trials',
    **{i: _NO_CHANGE for i in (2, 6, 47, 68, 70, 72, 101, 131, 142)},
    46: 'garbled ("the same number three has been drawn for the past few times")',
    128: 'garbled ("is your next bet more likely to be odd?")',
}


# ----------------------------------------------------------------------
# independence_risk
# ----------------------------------------------------------------------
def swap_lottery_order(text: str) -> Optional[str]:
    """Describe the two lotteries in the other order. Every template reads '... {p}% <preferred item> ... {q}% <c>
    ... {p}% <other item> ... {q}% <c> ...' with the labels Option {first} / Option {second}; the two '{p} ... {q}'
    segments are exchanged and, when the labels sit outside them, so are the labels, so that Option {first} (the key)
    still names the lottery with the preferred item, now described second."""
    ps = [m.start() for m in re.finditer(r'\{p\}', text)]
    qs = [m.start() for m in re.finditer(r'\{q\}', text)]
    if len(ps) != 2 or len(qs) != 2 or not ps[0] < qs[0] < ps[1] < qs[1]:
        return None
    s1, s2 = text[ps[0]:qs[0]], text[ps[1]:qs[1]]
    out = text[:ps[0]] + s2 + text[qs[0]:ps[1]] + s1 + text[qs[1]:]
    inside = [('{first}' in s) or ('{second}' in s) for s in (s1, s2)]
    if any(inside):
        return out if ('{first}' in s1 and '{second}' in s2) else None
    if out.count('{first}') != 1 or out.count('{second}') != 1:
        return None
    return out.replace('{first}', '\0').replace('{second}', '{first}').replace('\0', '{second}')

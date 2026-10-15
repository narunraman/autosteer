"""
Load-time template corrections for consumer-theory elements (review, October 2026).

Kept apart from template_fixes.py so that the rules for one group of elements read together; `install` chains
each rewrite after any rewrite template_fixes already has for the element (a rewrite returns the new question text,
or None to drop the template).

- deriving_demand: keep only templates that say which good is x and which is y, drop those that define x or y as
  money spent (the key is a quantity, coefficient / price), and say what p_x and p_y are.
- marginal_utility, mrs_utility: state the answer format ((MU_x, MU_y); the ordered pair (MU_x/MU_y, MU_y/MU_x)).
- aggregation_of_consumer_demand, aggregation_of_offer_curve: each function is per member of its group; round to
  two decimals.
- law_of_demand: other demand determinants unchanged (a movement along one downward-sloping demand curve), one
  option consistent with the law; templates that describe a demand shift (popularity, trends, ...), a quantity that
  is not the quantity demanded (stock, amount available, supplied, performed, sold, "we use", or never called
  demand), a "price" that moves against the price (yield, interest rate, return) or that state the direction of the
  asked value are dropped, and so are those whose final question asks for something else than the new value with a
  number in it (otherwise the final question is replaced);
- deriving_hicksian_demand: options rounded to two decimals (stated); templates whose goods are money spent or whose
  "price" is a payment received (earnings per work hour) are dropped.
- diminishing_marginal_utility: the final question asks how the marginal utility of the varied good (a derivative)
  changes between the two quantities (many asked about overall satisfaction or "each additional dose").
- price_elasticity: templates whose quantity need not be the quantity demanded (distributed, produced, supplied, in
  stock, available) are dropped. "change in the quantity demanded" (the key is the new level) and
  "increase in demand" (a movement along the curve) and "the likely cause" (the options are prices) reworded.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import pandas as pd

_SENTENCE_START = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def insert_before_last_sentence(text: str, sentence: str) -> str:
    text = text.rstrip()
    starts = [m.end() for m in _SENTENCE_START.finditer(text)]
    if not starts:
        return sentence + ' ' + text
    cut = starts[-1]
    return text[:cut] + sentence + ' ' + text[cut:]


def last_sentence(text: str) -> tuple[int, str]:
    text = text.rstrip()
    starts = [m.end() for m in _SENTENCE_START.finditer(text)]
    cut = starts[-1] if starts else 0
    return cut, text[cut:]


# ---------------------------------------------------------------------------------------------------------------
# deriving_demand
# ---------------------------------------------------------------------------------------------------------------
_VERB = r'(?:is|are|represents?|denotes?|stands? for|refers? to|signif(?:y|ies)|indicates?|measures?|captures?|for)'


def names_good(text: str, v: str) -> bool:
    """The text says what the variable v (x or y) stands for."""
    return bool(re.search(rf"\({v}\)|'{v}'|\b{v} {_VERB}\b|(?:denoted|represented|labell?ed|indicated) (?:by|as) "
                          rf"'?{v}\b|\b{v} (?:units|hours|amount)|\bby '?{v}\b", text))


# x or y defined as an amount of money ("y represents the amount of money spent on online courses",
# "If the expenditure on online courses is denoted by x"): the key coefficient / p_y is then not that amount
_MONEY = re.compile(r"\b[xy]'? (?:\w+ ){0,2}(?:the )?(?:amount of money|amount spent|money|spending|expenditure|dollars|budget "
                    r"share)\b|(?:money|spending|expenditure|dollars) (?:spent )?on [\w\- ]+ (?:is |are )?"
                    r"(?:denoted |represented )?(?:by |as )?\(?[xy]\)?\b", re.I)
_DD_ASKED = re.compile(r'demand function[^?]*? for ([^?]+)\?')
DERIVING_DEMAND_PRICES = 'Here p_x and p_y are the prices of one unit of x and of y.'


def deriving_demand_rewrite(row: pd.Series) -> Optional[str]:
    text = row['question_text']
    if not (names_good(text, 'x') and names_good(text, 'y')):
        return None
    m = _DD_ASKED.search(text)
    if _MONEY.search(text) or (m and re.search(r'money|spending|expenditure', m.group(1), re.I)):
        return None
    return insert_before_last_sentence(text, DERIVING_DEMAND_PRICES)


# ---------------------------------------------------------------------------------------------------------------
# marginal_utility, mrs_utility
# ---------------------------------------------------------------------------------------------------------------
MARGINAL_UTILITY_FORMAT = ('Answer with the pair (MU_x, MU_y), the partial derivatives of U with respect to x and '
                           'to y.')
MRS_FORMAT = ('Answer with the ordered pair (MU_x/MU_y, MU_y/MU_x), where MU_x and MU_y are the marginal utilities '
              'of x and y (x, y > 0).')


def _append(sentence: str) -> Callable[[pd.Series], Optional[str]]:
    def rewrite(row: pd.Series) -> Optional[str]:
        return row['question_text'].rstrip() + ' ' + sentence
    return rewrite


# ---------------------------------------------------------------------------------------------------------------
# aggregation_of_consumer_demand, aggregation_of_offer_curve
# ---------------------------------------------------------------------------------------------------------------
AGG_DEMAND_CONVENTION = ('Each demand function gives the quantity demanded by a single member of its group at price '
                         'P; the answer is the total over both groups, rounded to two decimal places.')
AGG_OFFER_CONVENTION = ('Each offer curve gives the quantity offered by a single member of its group at price P; '
                        'the answer is the total over both groups (numbers rounded to two decimal places).')


def _insert(sentence: str) -> Callable[[pd.Series], Optional[str]]:
    def rewrite(row: pd.Series) -> Optional[str]:
        return insert_before_last_sentence(row['question_text'], sentence)
    return rewrite


# ---------------------------------------------------------------------------------------------------------------
# law_of_demand
# ---------------------------------------------------------------------------------------------------------------
LAW_OF_DEMAND_CONVENTION = ('Assume nothing else that affects demand has changed (incomes, tastes, the prices of '
                            'other goods), so the change is a movement along an unchanged, downward-sloping demand '
                            'curve; exactly one of the values below is consistent with this.')
# a shift of demand: popularity, trends, rushes, advertising, income changes
_DEMAND_SHIFT = re.compile(r'popularity|(?:became|become|becoming|more|less|increasingly|growing|gaining) popular|'
                           r'\btrend|craze|\brush\b|\bfad\b|hype|advertis|\bincomes?\b|recession|\bboom\b|'
                           r'\bseason(?:al)?\b|word of mouth|endorse', re.I)
_DIR = (r'(?:risen|rose|rises|rising|increas\w*|gone up|went up|goes up|dropp\w*|drops|fallen|fell|falls|falling|'
        r'decreas\w*|declin\w*|raised|raising|raises|lower(?:ed|ing|s)?|reduc\w*|cuts?|cutting|hike\w*|slash\w*|'
        r'discount\w*|mark(?:ed)? up|mark(?:ed)? down)')
# the direction of the price stated in a question that asks for the new price
_PRICE_DIRECTION = re.compile(
    rf"\bprice\b(?:(?!demand|quantit|sold|sell|buy|bought|purchas|\{{)[^.?]){{0,30}}\b{_DIR}\b|"
    rf"\b{_DIR} (?:the |its |their |his |her |my |our )?(?:\w+ )?prices?\b|"
    r"\bprice (?:increase|decrease|reduction|cut|hike|drop|rise|discount|markdown|mark-up)s?\b|"
    r"\b(?:higher|lower|cheaper|more expensive)\b", re.I)
# the direction of the quantity stated in a question that asks for the new quantity
_QUANTITY_DIRECTION = re.compile(
    rf"\b(?:quantity|demand|sales|number)\b(?:(?!price|\{{)[^.?]){{0,40}}\b(?:will|would|might|could|to) "
    rf"(?:likely )?(?:increase|decrease|rise|fall|drop|go up|go down|decline|grow)\b", re.I)
_CHANGE_IN_Q = re.compile(r'\b(?:scenario for )?(?:the )?(?:probable |likely )?change in (?:the )?quantity demanded',
                          re.I)
_DEMAND_AS_QUANTITY = [
    (re.compile(r'\b(increase|rise|growth|decrease|drop|fall|decline|reduction|change|shift) in (?:the )?demand\b',
                re.I), r'\1 in the quantity demanded'),
    (re.compile(r'\b(?:the )?demand for ([^.,?{]+?) (has |had |then )?(increased|decreased|rose|fell|grew|dropped|'
                r'drops|rises|increases|decreases|declined|declines|went up|went down|goes up|goes down)\b', re.I),
     r'the quantity of \1 demanded \2\3'),
    (re.compile(r'\b(?:the )?demand (has |had |then )?(increased|decreased|rose|fell|grew|dropped|drops|rises|'
                r'increases|decreases|declined|declines|went up|went down|goes up|goes down)\b', re.I),
     r'the quantity demanded \1\2'),
    (re.compile(r'\b(increased|decreased|higher|lower|reduced|greater|initial|new|original) demand\b', re.I),
     r'\1 quantity demanded'),
    (re.compile(r'\b(?:an? |the |this |such an? )?demand (increase|decrease|reduction|drop|rise|decline|change)s?\b',
                re.I), r'this \1 in the quantity demanded'),
    (re.compile(r'\b(?:an? |the )?demand (was|is|of|at|stood at|level (?:was|of|is))(?= (?:\w+ )?\{(?:start|end)_quantity\})',
                re.I), r'the quantity demanded \1'),
]


_ASKS_PRICE = re.compile(r'\b(?:what|which)\b[^?]*\bprice\b', re.I)
_ASKS_QUANTITY = re.compile(r'\b(?:new |the |expected |likely |probable )quantity\b|\bhow many\b|\bnumber of\b|\bunits\b',
                            re.I)
_ASKS_OTHER = re.compile(r'\b(?:reason|happen\w*|infer\w*|explain\w*|explanation|why|factor|driv\w+|outcome|'
                         r'decision|behaviou?r|direction)\b|price change|change in (?!the quantity demanded)|'
                         r'\bhow\b(?! many)', re.I)


# a quantity that is not the quantity demanded (stock on hand, amount available, supplied, performed)
_NOT_DEMANDED = re.compile(r'\bin stock\b|\bthe stock (?:has )?(?:decreased|increased|fell|rose|dropped|went)|'
                           r'\bavailab\w*|\binventor\w*|\bsuppl(?:y|ied|ies)\b|\bproduc(?:e|ed|es|tion)\b|'
                           r'\bperform\w*', re.I)


# round 2 of the review: a "price" that moves against the price (a bond's yield, an interest rate, a return), and
# stories whose quantity is never called demand ("the quantity we use", "quantity sold", "purchased by investors")
_INVERSE_PRICE = re.compile(r'\byield|interest rate|\breturns?\b|discount rate|\bcoupon', re.I)
_DEMAND_WORD = re.compile(r'\bdemand(?:ed)?\b', re.I)
_SOLD = re.compile(r'\bsold\b|\bsales\b|\bsell(?:s|ing)?\b|\bwe use\b', re.I)


def law_of_demand_rewrite(row: pd.Series) -> Optional[str]:
    text = row['question_text']
    typ = str(row['type'])
    if _DEMAND_SHIFT.search(text) or _NOT_DEMANDED.search(text) or _INVERSE_PRICE.search(text) \
            or _SOLD.search(text) or not _DEMAND_WORD.search(text):
        return None
    asks_price = typ.startswith('quantity')
    if asks_price:
        if _PRICE_DIRECTION.search(text):
            return None
        for pat, rep in _DEMAND_AS_QUANTITY:
            text = pat.sub(rep, text)
    else:
        if _QUANTITY_DIRECTION.search(text):
            return None
        text = _CHANGE_IN_Q.sub(lambda m: ('the ' if m.group(0).lower().startswith('the') else '')
                                + 'new quantity demanded', text)
        text = re.sub(r'\bthe (most )?(likely|probable) the new\b', r'the \1\2 new', text)
    # the options are new prices / new quantities: the final question must ask for one
    cut, last = last_sentence(text)
    asks = _ASKS_PRICE if asks_price else _ASKS_QUANTITY
    if not asks.search(last) or _ASKS_OTHER.search(last):
        if '{' in last:
            return None
        text = text[:cut] + ('Which new price is most consistent with this change?' if asks_price
                             else 'What is the most likely new quantity demanded?')
    return insert_before_last_sentence(text, LAW_OF_DEMAND_CONVENTION)


# ---------------------------------------------------------------------------------------------------------------
# deriving_hicksian_demand
# ---------------------------------------------------------------------------------------------------------------
HICKSIAN_ROUNDING = 'Numbers in the options are rounded to two decimal places.'
# a good defined as money spent ("x1 represents spending on cloud infrastructure"): the key is a quantity
_HICKS_MONEY = re.compile(r"\b(?:x|y|z|x_?[123])'? (?:\w+ ){0,2}(?:the )?(?:amount of money|amount spent|money|"
                          r"spending|expenditure|dollars|budget)\b|(?:money|spending|expenditure|dollars) (?:spent )?"
                          r"on [\w\- ]+ (?:is |are )?(?:denoted |represented )?(?:by |as )?\(?(?:x|y|z|x_?[123])\)?\b",
                          re.I)
# a "price" that is a payment received (earnings per work hour, a wage): the expenditure is then unbounded below
_HICKS_EARNINGS = re.compile(r'\bearn\w*|\bwages?\b|\bsalar\w*|\bincome (?:per|from)\b|\brevenue|\bprofit|'
                             r'\bpaid (?:per|for each)\b|\breturn per\b', re.I)


def deriving_hicksian_demand_rewrite(row: pd.Series) -> Optional[str]:
    text = row['question_text']
    if _HICKS_MONEY.search(text) or _HICKS_EARNINGS.search(text):
        return None
    return insert_before_last_sentence(text, HICKSIAN_ROUNDING)


# ---------------------------------------------------------------------------------------------------------------
# diminishing_marginal_utility
# ---------------------------------------------------------------------------------------------------------------
def diminishing_marginal_utility_rewrite(row: pd.Series) -> Optional[str]:
    """The options describe the marginal utility of the varied good (a derivative, evaluated at the start and end
    quantities); many final questions asked about overall satisfaction or "each additional dose" (a discrete
    increment). The final question is replaced by one that asks for the marginal utility as a derivative; templates
    that do not say which good is x and which is y, or whose final sentence carries the only mention of another
    value, are dropped."""
    text = row['question_text'].rstrip()
    if not (names_good(text, 'x') and names_good(text, 'y')):
        return None   # which good is held fixed at {fixed} must be clear
    var = 'y' if str(row['type']).startswith('x_fixed') else 'x'
    cut, last = last_sentence(text)
    head = text[:cut]
    for ph in ('{utility_func}', '{fixed}'):
        if ph in last and ph not in head:
            return None
    return (head + f"Treating quantities as continuous, how does the marginal utility of {var} (the partial "
            f"derivative of the utility function with respect to {var}) change as {var} rises from {{start}} to "
            f"{{end}}?")


# ---------------------------------------------------------------------------------------------------------------
# price_elasticity
# ---------------------------------------------------------------------------------------------------------------
# a quantity that need not be the quantity demanded ("political flyers distributed", produced, in stock)
_PE_NOT_DEMANDED = re.compile(r'\bdistribut\w*|\bproduc(?:e|ed|es|tion)\b|\bsuppl(?:y|ied|ies)\b|\bin stock\b|'
                              r'\binventor\w*|\bavailab\w*', re.I)


def price_elasticity_rewrite(row: pd.Series) -> Optional[str]:
    return None if _PE_NOT_DEMANDED.search(row['question_text']) else row['question_text']


# ---------------------------------------------------------------------------------------------------------------
REWRITES: dict[str, Callable[[pd.Series], Optional[str]]] = {
    'deriving_demand': deriving_demand_rewrite,
    'marginal_utility': _append(MARGINAL_UTILITY_FORMAT),
    'mrs_utility': _append(MRS_FORMAT),
    'aggregation_of_consumer_demand': _insert(AGG_DEMAND_CONVENTION),
    'aggregation_of_offer_curve': _insert(AGG_OFFER_CONVENTION),
    'law_of_demand': law_of_demand_rewrite,
    'deriving_hicksian_demand': deriving_hicksian_demand_rewrite,
    'diminishing_marginal_utility': diminishing_marginal_utility_rewrite,
    'price_elasticity': price_elasticity_rewrite,
}


def chain(first: Optional[Callable], second: Callable) -> Callable:
    if first is None:
        return second

    def both(row: pd.Series) -> Optional[str]:
        text = first(row)
        if text is None:
            return None
        row = row.copy()
        row['question_text'] = text
        return second(row)
    return both


def install(rewrites: dict) -> None:
    for element, fn in REWRITES.items():
        rewrites[element] = chain(rewrites.get(element), fn)

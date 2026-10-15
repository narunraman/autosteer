"""
Load-time corrections applied to template datasets before sampling.

The committed templates (data/templates/<element>/) are
LLM-written and a few of them are inconsistent with what their filler
computes. Rather than rewriting the pickles, the sampler applies the small,
documented corrections below every time a template is loaded, so the fix is
visible in code review and the raw templates stay untouched.

Four kinds of correction are supported, all keyed by element name:

- TYPE_EXPANSIONS: clone templates of one type into other types. This is
  what the old `type_expansion` notebook helper did when the published
  datasets were built, but it was never saved into the committed templates.
- TYPE_RENAMES: relabel a type whose name does not describe what the filler
  generates.
- TEMPLATE_FILTERS: drop individual templates whose wording contradicts the
  numbers the filler produces for their type.
- TEMPLATE_REWRITES: edit the question text of a template (or drop it by
  returning None) so it asks what the filler answers.

In addition, for every element, escaped placeholders ('{{price}}') are
unescaped, and templates that spell one of the element's placeholder names
without braces (e.g. "num_victims") are dropped.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import pandas as pd

# ----------------------------------------------------------------------
# Type expansions
# ----------------------------------------------------------------------

# element -> {source type substring: [replacement substrings]}
# Every template whose type contains the source substring is cloned once per
# replacement, with the substring replaced in its type.
TYPE_EXPANSIONS: dict[str, dict[str, list[str]]] = {
    # The committed templates only have linear_* types, for which the answer
    # is always "remains constant". The published STEER-ME data also had
    # cobb_douglas_* and quadratic_* types, cloned from the linear templates;
    # the filler already supports them. The templates do not name a function
    # family, so the cloned wording stays valid.
    'diminishing_marginal_product': {'linear': ['cobb_douglas', 'quadratic']},
}

# ----------------------------------------------------------------------
# Type renames
# ----------------------------------------------------------------------

TYPE_RENAMES: dict[str, dict[str, str]] = {
    # The filler always generates a quadratic supply function for this type.
    'supply_elasticity': {'Cobb_douglas_supply_given': 'Quadratic_supply_given'},
}

# ----------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------

_UP = re.compile(r'\b(increas\w*|ris\w*|rose|grow\w*|grew|jump\w*|climb\w*|surg\w*|goes up|went up|go up|expand\w*|soar\w*|boost\w*)\s*$')
_DOWN = re.compile(r'\b(decreas\w*|drop\w*|fall\w*|fell|declin\w*|reduc\w*|goes down|went down|go down|shrink\w*|shrank|dip\w*|plummet\w*|slump\w*|diminish\w*|contract\w*)\s*$')


def quantity_direction(text: str) -> Optional[str]:
    """Return 'up'/'down' from the verb right before "from {start_quantity}", else None."""
    m = re.search(r'(.{0,60}?)\s*from\s+\{start_quantity\}', text)
    if not m:
        return None
    head = m.group(1).lower()
    if _UP.search(head):
        return 'up'
    if _DOWN.search(head):
        return 'down'
    return None


def _price_elasticity_filter(row: pd.Series) -> bool:
    # The filler follows the law of demand: price_increase -> quantity falls,
    # price_decrease -> quantity rises. About 90 price_decrease templates say
    # the quantity "decreases", which contradicts the numbers. Drop them.
    direction = quantity_direction(row['question_text'])
    if direction is None:
        return True
    if 'price_increase' in row['type']:
        return direction == 'down'
    if 'price_decrease' in row['type']:
        return direction == 'up'
    return True


def _supply_elasticity_filter(row: pd.Series) -> bool:
    # The filler shows quantity supplied as a function of price, Q = f(P).
    # About 10% of templates call it "supply function (or marginal cost
    # function)", i.e. an inverse supply P = MC(Q); drop those.
    return 'marginal cost' not in row['question_text'].lower()


def _family_word_filter(rules: list[tuple[str, str]]) -> Callable[[pd.Series], bool]:
    """Drop templates whose type contains `type_part` but whose text matches `pattern`."""
    compiled = [(t, re.compile(p, re.I)) for t, p in rules]

    def keep(row: pd.Series) -> bool:
        return not any(t in str(row['type']) and pat.search(row['question_text']) for t, pat in compiled)
    return keep


def _names_x_and_y(text: str) -> bool:
    """The text says which good is x and which is y ("pizza (x)", "'x' represents pizza")."""
    def named(v):
        return bool(re.search(rf'\({v}\)', text) or
                    re.search(rf"'?\b{v}\b'? (?:is|represents|denotes|stands for|refers to)\b", text))
    return named('x') and named('y')


TEMPLATE_FILTERS: dict[str, Callable[[pd.Series], bool]] = {
    # micro_utility: the options are bundles "x = 4, y = 7"; keep templates that say which good is x and y
    'micro_utility': lambda row: _names_x_and_y(row['question_text']),
    'price_elasticity': _price_elasticity_filter,
    'supply_elasticity': _supply_elasticity_filter,
    # Leftovers of cloning templates across types (the `type_expansion`
    # issue): the text names a function family the filler does not use.
    # quadratic_* types say "linear demand" in 2 templates.
    'find_eq_price': _family_word_filter([('quadratic', r'\blinear\b'), ('linear', r'quadratic')]),
    # the filler always uses a Cobb-Douglas utility; 9 templates say "linear utility"
    'deriving_hicksian_demand': _family_word_filter([('', r'\blinear\b')]),
    # both segments are linear; 1 template calls one "non-linear"
    'aggregation_of_capital_demand': _family_word_filter([('', r'non-?linear')]),
    # the capital-stock types were dropped (decision of 3 October 2026); see the tfp_shocks filler
    'tfp_shocks': lambda row: 'capital' not in str(row['type']),
    # the filler only provides {price} for intersect_given; some example
    # templates of the other types use it and cannot be filled
    'efficient_surplus': lambda row: row['type'] == 'intersect_given' or '{price}' not in row['question_text'],
    # review: no externality markets (pollution credits, water rights) and no stories without goods to trade
    'welfare_theorem_1': lambda row: _wt_keep(row),
    'welfare_theorem_2': lambda row: _wt_keep(row),
}

# ----------------------------------------------------------------------
# Rewrites
# ----------------------------------------------------------------------

_ASK = re.compile(r'(demand function(?: that \w+ \w+| I have| \w+ has| you have| the person has| they have| my)? for )([^?]+)(\?)')
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


def _deriving_demand_rewrite(row: pd.Series) -> Optional[str]:
    # Every good_2 template was cloned from good_1 and still asks for the
    # demand for the first good (x), while the filler answers for good y.
    # Rewrite the final question to name good y; drop the template if the
    # name of good y cannot be found in the text.
    text = row['question_text']
    if row['type'] != 'good_2':
        return text
    m = _ASK.search(text)
    y = y_good_name(text)
    if not m or not y:
        return None
    asked = m.group(2).strip()
    # keep a leading qualifier such as "spending on" if y does not carry one
    prefix = ''
    q = re.match(r'^((?:the )?(?:consumption|spending|expenditure|quantity|amount) (?:of|on) )', asked, re.I)
    if q and not re.match(r'^(consumption|spending|expenditure|quantity|amount|money|time|hours)\b', y, re.I):
        prefix = q.group(1)
    if _strip_article(y).lower() in _strip_article(asked).lower():
        return text  # already asks for good y
    return text[:m.start(2)] + prefix + y + text[m.end(2):]


def _intertemporal_rewrite(row: pd.Series) -> Optional[str]:
    # The 0.9-0.99 parameter is used as the discount factor beta (Narun's
    # default); the templates call it a "discount rate". Templates that do
    # not label {gamma} as a risk-aversion/curvature coefficient (e.g.
    # "social discount factor is {gamma}") are dropped: the problem is then
    # not identified.
    text = row['question_text']
    if not re.search(r'(risk|gamma|curvature)[^.{]{0,70}\{gamma\}', text, re.I):
        return None
    text = re.sub(r'discount rate', 'discount factor', text, flags=re.I)
    text = re.sub(r'at an? (?:annual )?rate of (\{discount_rate\})', r'with a discount factor of \1', text)
    # any remaining "rate" right before the value, e.g. "rate is {discount_rate}"
    i = text.find('{discount_rate}')
    if i >= 0:
        head = text[max(0, i - 40):i]
        j = head.rfind(' rate')
        if j >= 0 and 'factor' not in head[j:]:
            k = max(0, i - 40) + j
            text = text[:k] + ' factor' + text[k + 5:]
    return text


def _segment_count_rewrite(row: pd.Series) -> Optional[str]:
    # Aggregation templates name a total ({num_markets}/{num_firms}/...) and
    # the two segment sizes {num_a}, {num_b}. A few use the total where the
    # first segment's size belongs ("Across {num_markets} clinics, the demand
    # is {demand_function_a}. In the remaining {num_b} ..."): use {num_a}.
    text = row['question_text']
    if '{num_a}' in text:
        return text
    fa = re.search(r'\{\w+_a\}', text)
    if not fa:
        return text
    totals = list(re.finditer(r'\{num_(?:markets|firms|consumers)\}', text[:fa.start()]))
    if not totals:
        return None  # segment size of group a is never stated
    t = totals[-1]
    return text[:t.start()] + '{num_a}' + text[t.end():]


_TFP_ASSUMPTION = 'Assume a competitive market, so that in the long run the price of output equals the unit cost of production.'
# Abbreviations that end in a period but do not end a sentence ("Dr. Anna", "e.g. Mr. Smith"); a convention inserted
# "before the last sentence" went between "Dr." and the name (Re-review). Each lookbehind is fixed width.
_ABBREVIATIONS = ('Dr', 'Mr', 'Mrs', 'Ms', 'Mx', 'Prof', 'St', 'Jr', 'Sr', 'Inc', 'Co', 'Ltd', 'Corp', 'vs', 'etc',
                  'e.g', 'i.e', 'approx', 'No', 'Mt', 'Gen', 'Capt', 'Lt', 'Sgt', 'Rev', 'Hon')
_SENTENCE_START = re.compile(''.join(rf'(?<!\b{re.escape(a)}\.)' for a in _ABBREVIATIONS)
                             + r'(?<=[.!?])\s+(?=[A-Z])')


def _tfp_rewrite(row: pd.Series) -> Optional[str]:
    # The key is the percent change in the long-run competitive price (= unit cost); say so before the final
    # question. Templates that do not end in a question (4) are dropped.
    text = row['question_text'].strip()
    if not text.endswith('?'):
        return None
    starts = [m.end() for m in _SENTENCE_START.finditer(text)]
    if not starts:
        return None
    k = starts[-1]
    return text[:k] + _TFP_ASSUMPTION + ' ' + text[k:]


# Quality audit (October 2026): wording fixes of the GPT-written templates.

# "the cost of labor per hour is {c_function}" printed "8.47L" (a cost function) where a wage belongs; the fillers
# now also give the wage alone. "per unit of output" phrasings would call the wage a unit cost: dropped.
_PER_HOUR_COST = re.compile(r"((?:per (?:[\w-]+ )?(?:hour|unit)\b(?: of [\w\- ]+?)?|hourly [\w-]+|wages?|rate)"
                            r"\b[^.,;{]{0,40}?)\{c_function\}", re.I)


def _per_hour_cost_rewrite(row: pd.Series) -> Optional[str]:
    text = row['question_text']
    m = _PER_HOUR_COST.search(text)
    if not m:
        return text
    if re.search(r'per unit of output', m.group(1), re.I):
        return None
    return text[:m.start()] + m.group(1) + '${wage}' + text[m.end():]


def _dmp_capital_rewrite(row: pd.Series) -> Optional[str]:
    """The production functions are written in L and K, the templates define 'C' as capital."""
    return re.sub(r'\bC\b', 'K', row['question_text'])


def _sentence_space_rewrite(row: pd.Series) -> Optional[str]:
    """'dividends.What is' -> 'dividends. What is'; 'You are the row player What is' -> '... player. What is'."""
    text = re.sub(r'(?<=[a-z])([.?!])(?=[A-Z][a-z])', r'\1 ', row['question_text'])
    return re.sub(r'\b(row|column) player (?=(What|Which|How)\b)', r'\1 player. ', text)


def _ballots_rewrite(row: pd.Series) -> Optional[str]:
    """The ballot lines "- 95 voters voted for Q > Z > S" start on a line of their own, and the question after them
    too (it went on the line of the last ballot)."""
    text = re.sub(r'(?<=:) *\{ballots\}', '\n{ballots}', row['question_text'])
    return re.sub(r'\{ballots\} +(?=[A-Z])', '{ballots}\n', text)


def _price_elasticity_method_rewrite(row: pd.Series) -> Optional[str]:
    """State the convention before the final question (the simple and the midpoint formula give different keys)."""
    from autosteer.filler_tools.comparative_statics_demand_tools import PRICE_ELASTICITY_METHOD
    text = row['question_text'].rstrip()
    starts = [m.end() for m in _SENTENCE_START.finditer(text)]
    if not starts:
        return text + ' ' + PRICE_ELASTICITY_METHOD
    cut = starts[-1]
    return text[:cut] + PRICE_ELASTICITY_METHOD + ' ' + text[cut:]


_ITERATED_PART2 = {
    'Which action should your opponent not play?':
        'Your opponent knows that you will not play your dominated action. Which action should your opponent not '
        'play in the remaining game?',
    "Let's say your opponent also knows which of your actions is dominated. Which action should your opponent not play?":
        "Let's say your opponent also knows which of your actions is dominated, and that you will not play it. Which "
        "action should your opponent not play in the remaining game?",
    'Which action should you not play?':
        'You know that your opponent will not play their dominated action. Which action should you not play in the '
        'remaining game?',
    'Knowing that your opponent has a dominated action, which action should you not play?':
        'Knowing that your opponent has a dominated action and will not play it, which action should you not play in '
        'the remaining game?',
}


def _iterated_part2_rewrite(row: pd.Series) -> Optional[str]:
    """iterated_removal part 2 without the reduced table: say that the first dominated action is removed (the bare
    "Which action should you not play?" did not; review, October 2026)."""
    text = row['question_text']
    return _ITERATED_PART2.get(text.strip(), text)


_BB_EXACT = [
    (re.compile(r'Does the mechanism collect exactly the cost of ([^?]+)\?'),
     r'How does the money the mechanism collects compare with the cost of \1?'),
    (re.compile(r'Does this mechanism collect exactly what ([^?]+?) costs\?'),
     r'How does what this mechanism collects compare with what \1 costs?'),
]


def _budget_balance_ask_rewrite(row: pd.Series) -> Optional[str]:
    """"Does the mechanism collect exactly the cost?" asked a yes/no question while the options classify the budget
    (and under the weak definition a surplus is "balanced"); ask for the comparison instead (review, Oct 2026)."""
    text = row['question_text']
    for pat, rep in _BB_EXACT:
        text = pat.sub(rep, text)
    return text


def _subgame_rewrite(row: pd.Series) -> Optional[str]:
    return row['question_text'].replace("opponents's", "opponent's")


# Review (October 2026): conventions the keys rely on, stated before the final question.

def before_last_sentence(text: str, sentence: str) -> str:
    """Insert `sentence` before the last sentence of `text` (at the end if it has only one sentence)."""
    text = text.strip()
    starts = [m.end() for m in _SENTENCE_START.finditer(text)]
    if not text.endswith(('?', '.', '!', ':')):   # a few templates lose the final question mark
        last = text[starts[-1]:] if starts else text
        text += '?' if re.search(r'\b(what|how|which|who|is|are|does|do|can|should|would|could|will)\b', last,
                                 re.I) else '.'
    if not starts:
        return text + ' ' + sentence
    # before the last sentence that asks something (a trailing fragment such as "Robert's total utility." is not it)
    bounds = [0] + starts + [len(text)]
    cut = starts[-1]
    for i in range(len(bounds) - 2, 0, -1):
        if _ASKS.search(text[bounds[i]:bounds[i + 1]]):
            cut = bounds[i]
            break
    return text[:cut] + sentence + ' ' + text[cut:]


_ASKS = re.compile(r'\?|\b(what|how|which|calculate|determine|find|compute|work out|estimate)\b', re.I)


CONSUMER_SURPLUS_CONVENTION = ('The demand function gives the price P buyers are willing to pay as a function of the '
                               'quantity Q (an inverse demand curve), and the good is treated as perfectly divisible.')
DWL_CONVENTION = ('Here both curves give the price P as a function of the quantity Q, the supply curve is the '
                  "seller's marginal cost curve, and the seller charges one price for every unit. The deadweight loss "
                  'is the total surplus lost relative to the competitive quantity, where demand equals marginal cost.')
DWL_PRICE_EXACT = 'Take the stated monopoly price as exact.'
EFFICIENT_SURPLUS_CONVENTION = 'Here both functions give the price P as a function of the quantity Q.'
EFFICIENT_SURPLUS_PRICE_ROUNDED = ('The stated equilibrium price is rounded to two decimal places; use the exact '
                                   'intersection of the two curves.')


def _consumer_surplus_rewrite(row: pd.Series) -> Optional[str]:
    return before_last_sentence(row['question_text'], CONSUMER_SURPLUS_CONVENTION)


def _deadweight_loss_rewrite(row: pd.Series) -> Optional[str]:
    extra = DWL_CONVENTION + (' ' + DWL_PRICE_EXACT if '{m_price}' in row['question_text'] else '')
    return before_last_sentence(row['question_text'], extra)


def _efficient_surplus_rewrite(row: pd.Series) -> Optional[str]:
    extra = EFFICIENT_SURPLUS_CONVENTION + (' ' + EFFICIENT_SURPLUS_PRICE_ROUNDED
                                            if '{price}' in row['question_text'] else '')
    return before_last_sentence(row['question_text'], extra)


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
_EQUALLY_LIKELY = re.compile(r'\b(?:are|is|being) equally likely,? (?:with|having) probabilit', re.I)


def _state_based_rewrite(row: pd.Series) -> Optional[str]:
    text = _EQUALLY_LIKELY.sub('occur with probabilit', row['question_text'])
    if re.search(r'equally likely', text, re.I):
        return None   # stated probabilities are not equal
    return before_last_sentence(text, STATE_BASED_CONVENTION)


def _intertemporal_full_rewrite(row: pd.Series) -> Optional[str]:
    text = _intertemporal_rewrite(row)
    return None if text is None else before_last_sentence(text, INTERTEMPORAL_CONVENTION)


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


def _wt_keep(row: pd.Series) -> bool:
    text = row['question_text']
    return not _WT_EXTERNALITY.search(text) and not _WT_NO_GOODS.search(text)


WT1_FURTHER = ('After the market reaches this equilibrium, is there any further trade that would make some '
               'participant better off without making anyone worse off?')


def _wt1_rewrite(row: pd.Series) -> Optional[str]:
    text = row['question_text']
    if row['type'] == 'further_trades':
        # re-review: "do I need to make any additional trades to improve my situation?" is not a question about
        # Pareto improvements; one final question for the type
        text = _replace_last_question(text, WT1_FURTHER)
        if text is None or text.strip() == WT1_FURTHER:
            return None
    return before_last_sentence(text, WT1_PREMISE)


def _wt2_rewrite(row: pd.Series) -> Optional[str]:
    new = WT2_WITHOUT if row['type'] == 'without_transfers' else WT2_WITH
    text = _replace_last_question(row['question_text'], new)
    if text is None or text.strip() == new:
        return None   # the whole template was the question: no story left
    return before_last_sentence(text, WT2_PREMISE)
# Review (October 2026), STEER logic / arithmetic / probability elements. The conventions live with their
# fillers, which strip them again where they look templates up by text (imported lazily: the fillers import the data
# directory's label tables).
def _add_sub_filter(row: pd.Series) -> bool:
    from autosteer.filler_tools.arithmetic_tools import keep_add_sub_template
    return keep_add_sub_template(row['question_text'])


def _mult_div_rounding_rewrite(row: pd.Series) -> Optional[str]:
    from autosteer.filler_tools.arithmetic_tools import with_rounding_note
    return with_rounding_note(row['question_text'])


def _bayes_rounding_rewrite(row: pd.Series) -> Optional[str]:
    from autosteer.filler_tools.probability_tools import BAYES_NOTE
    text = row['question_text'].rstrip()
    return text if text.endswith(BAYES_NOTE.strip()) else text + BAYES_NOTE


def _expectations_fair_rewrite(row: pd.Series) -> Optional[str]:
    from autosteer.filler_tools.expectation_tools import fairness_note
    return fairness_note(row['question_text'])


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


def _complement_rewrite(row: pd.Series) -> Optional[str]:
    text = _sentence_space_rewrite(row)
    for old, new in COMPLEMENT_REWORDS:
        text = text.replace(old, new)
    return text


TEMPLATE_FILTERS['add_sub'] = _add_sub_filter


TEMPLATE_REWRITES: dict[str, Callable[[pd.Series], Optional[str]]] = {
    'marginal_cost': _per_hour_cost_rewrite,
    'average_cost': _per_hour_cost_rewrite,
    'diminishing_marginal_product': _dmp_capital_rewrite,
    'complement_rule': _complement_rewrite,
    'interpret_games': _sentence_space_rewrite,
    'subgame_nash': _subgame_rewrite,
    'iterated_removal': _iterated_part2_rewrite,
    'budget_balance': _budget_balance_ask_rewrite,
    'plurality_voting': _ballots_rewrite,
    'price_elasticity': _price_elasticity_method_rewrite,
    'tfp_shocks': _tfp_rewrite,
    'aggregation_of_capital_demand': _segment_count_rewrite,
    'aggregation_of_offer_curve': _segment_count_rewrite,
    'aggregation_of_labor_demand': _segment_count_rewrite,
    'aggregation_of_labor_supply': _segment_count_rewrite,
    'aggregation_of_consumer_demand': _segment_count_rewrite,
    'deriving_demand': _deriving_demand_rewrite,
    'intertemporal_consumption_smoothing': _intertemporal_full_rewrite,
    'consumer_surplus': _consumer_surplus_rewrite,
    'deadweight_loss': _deadweight_loss_rewrite,
    'efficient_surplus': _efficient_surplus_rewrite,
    'state_based_consumption': _state_based_rewrite,
    'welfare_theorem_1': _wt1_rewrite,
    'welfare_theorem_2': _wt2_rewrite,
    'mult_div': _mult_div_rounding_rewrite,
    'bayes_rule': _bayes_rounding_rewrite,
    'compute_expectations': _expectations_fair_rewrite,
}

# Review (October 2026), production elements: conventions the keys rely on, stated in the question.


def _with_text(row: pd.Series, text: str) -> pd.Series:
    r = row.copy()
    r['question_text'] = text
    return r


def _before_last_sentence(text: str, sentence: str) -> str:
    text = text.strip()
    starts = [m.end() for m in _SENTENCE_START.finditer(text)]
    if not starts:
        return sentence + ' ' + text
    k = starts[-1]
    return text[:k] + sentence + ' ' + text[k:]


_COST_CONVENTION = ('Take Q as the output given by the production function, L as its variable input and K as fixed at '
                    '{capital}; the variable cost is ${wage} per unit of L ({c_function} in total) and the fixed cost '
                    'is {f_cost}.')
_MC_ASK = ('Treat output as divisible and give the marginal cost dC/dQ as a function of Q, with the coefficient and '
           'the exponent rounded to two decimal places.')
_AC_ASK = 'Give the average cost C(Q)/Q, with the coefficient and the exponent rounded to two decimal places.'


def _cost_convention(ask: str):
    """marginal_cost / average_cost: the stories call the cost '4.25L' a cost per student, per solution or per user,
    leave L or Q undefined, and the keys round coefficient and exponent unannounced. State the model and the
    convention after the question (the final question usually carries the cost)."""
    def rewrite(row):
        text = _per_hour_cost_rewrite(row)
        if text is None:
            return None
        return text.rstrip() + ' ' + _COST_CONVENTION + ' ' + ask
    return rewrite


_RTS_CONVENTION = ('Here L and K are the two inputs of the production function, and returns to scale describe what '
                   'happens to output when both are increased by the same proportion.')


def _rts_rewrite(row: pd.Series) -> Optional[str]:
    """returns_to_scale: some stories name one input (the amount of wax) while the function has L and K."""
    return _before_last_sentence(row['question_text'], _RTS_CONVENTION)


_PROFIT_CONVENTION = 'Ignore the cost of capital and any other fixed costs: profit is sales revenue minus the cost of labor.'


def _profit_rewrite(row: pd.Series) -> Optional[str]:
    """profit_maximization: the profit key is revenue minus labor cost (no price of capital is given); say so."""
    if row['type'] != 'compute_profit':
        return row['question_text']
    return _before_last_sentence(row['question_text'], _PROFIT_CONVENTION)


_DMP_DEF = re.compile(r'\b([A-Z])\b(?=\s*,?\s*(?:is|represents|representing|denotes|denoting|stands for|standing for|'
                      r'refers to|indicates|signifies|measures|captures)\b|,\s+the\b)|\(([A-Z])\)')
_WH = re.compile(r'\b(how|what)\b', re.I)


def _dmp_rewrite(row: pd.Series) -> Optional[str]:
    """diminishing_marginal_product: the functions are in L and K, the stories define other letters (M for
    materials, T for tutors, F and A) and some ask about 'the average tuition outcome'. A lone other letter beside L
    (K) becomes K (L); a sentence says which of L and K increases from {start} to {end} and which is held at
    {fixed}; the final question asks how the marginal product of that input changes."""
    text = _dmp_capital_rewrite(row)
    var, other = ('L', 'K') if 'capital_fixed' in row['type'] else ('K', 'L')
    defined = {a or b for a, b in _DMP_DEF.findall(text)} - {'I', 'Q', 'Y'}
    extra = defined - {'L', 'K'}
    mapping = {}
    if len(extra) == 1:
        x = next(iter(extra))
        if 'L' in defined and 'K' not in defined:
            mapping[x] = 'K'
        elif 'K' in defined and 'L' not in defined:
            mapping[x] = 'L'
    elif len(extra) in (1, 2) and not defined & {'L', 'K'}:
        # neither L nor K is defined (F and A, S and T): the letter whose noun is in the clause that says what
        # increases from {start} is the varied input (and the other letter, if any, the fixed one)
        i = text.find('{start}')
        clause = re.split(r'[.!?]\s', text[:i])[-1].lower() if i >= 0 else ''
        hits = []
        for x in extra:
            m = re.search(rf'\b{x}\b\s*,?\s*(?:is|represents|representing|denotes|denoting|stands for|standing for|'
                          rf'refers to|indicates|signifies|measures|captures)\s+(?:the\s+)?(?:number of\s+|amount of\s+)?'
                          rf'([\w-]+)', text)
            if m and m.group(1).lower().rstrip('s') in clause:
                hits.append(x)
        if len(hits) == 1:
            mapping = {hits[0]: var}
            if len(extra) == 2:
                mapping[(extra - set(hits)).pop()] = other
    for x, y in mapping.items():
        if x == 'A':  # also an article: only where it is defined
            text = re.sub(r'\bA\b(?=\s*,?\s*(?:is|represents|representing|denotes|stands for|refers to)\b)', y, text)
        else:
            text = re.sub(rf'\b{x}\b', y, text)
    text = text.strip()
    starts = [m.end() for m in _SENTENCE_START.finditer(text)]
    k = starts[-1] if starts else 0
    last = text[k:]
    whs = list(_WH.finditer(last))
    ask = f'how does the marginal product of {var} change?'
    if whs and last[:whs[-1].start()].strip():
        last = last[:whs[-1].start()] + ask
    else:
        last = ask[0].upper() + ask[1:]
    role = (f'In the production function, {var} is the input that increases from {{start}} to {{end}}, and {other} is '
            f'the input held fixed at {{fixed}}.')
    return text[:k] + role + ' ' + last


def _agg_domain(noun: str):
    """aggregation_of_labor_demand / aggregation_of_capital_demand: linear demands are negative above their choke
    price, so the aggregate a - bP holds only where both are nonnegative; the demand_function type states that range
    ({p_max} from the filler), the numerical type says that the given price is P."""
    def rewrite(row):
        text = _segment_count_rewrite(row)
        if text is None:
            return None
        if row['type'] == 'demand_function':
            return _before_last_sentence(text, f'Here P is the {noun}; consider only values of P at which both '
                                               f'groups demand a nonnegative amount, 0 ≤ P ≤ {{p_max}}.')
        return re.sub(r'\$?\{price\}', '${price} (P = {price})', text, count=1)
    return rewrite


_UPWORD = re.compile(r'increas|rise|rais|boost|improv|grow|gain|enhanc|advanc|jump|surge|climb|up\b|higher|hike', re.I)
_DOWNWORD = re.compile(r'decreas|drop|fall|fell|declin|reduc|cut|lower|down\b|dip|loss|slump|diminish|shrink|worsen',
                       re.I)
_TFP_PCT = re.compile(r'\{(productivity|labor_price)_(increase|decrease)\}')


def _tfp_direction_rewrite(row: pd.Series) -> Optional[str]:
    """tfp_shocks: 'results in a {productivity_increase} in productivity' gave no direction; insert it where the
    clause has no direction word, drop the template where it states the opposite one. State what stays fixed (the
    other input price, and A for a wage shock) and that the answer is the percentage change in the output price."""
    text = row['question_text']
    m = _TFP_PCT.search(text)
    if m:
        head = re.split(r'[.!?]\s', text[:m.start()])[-1]
        tail = ' '.join(text[m.end():].split()[:4])
        window = head + ' ' + tail
        want, wrong = (_UPWORD, _DOWNWORD) if m.group(2) == 'increase' else (_DOWNWORD, _UPWORD)
        if not want.search(window):
            if wrong.search(window):
                return None
            text = text[:m.end()] + ' ' + m.group(2) + text[m.end():]
    text = _tfp_rewrite(_with_text(row, text))
    if text is None:
        return None
    fixed = ('the wage and the rental price of capital do not change' if 'productivity' in row['type']
             else 'productivity A and the rental price of capital do not change')
    return text.replace(_TFP_ASSUMPTION, _TFP_ASSUMPTION[:-1] + f', and that {fixed}. Answer with the percentage '
                                                                 f'change in the price of output.')


TEMPLATE_REWRITES.update({
    'marginal_cost': _cost_convention(_MC_ASK),
    'average_cost': _cost_convention(_AC_ASK),
    'returns_to_scale': _rts_rewrite,
    'profit_maximization': _profit_rewrite,
    'diminishing_marginal_product': _dmp_rewrite,
    'aggregation_of_labor_demand': _agg_domain('wage'),
    'aggregation_of_capital_demand': _agg_domain('price of capital'),
    'tfp_shocks': _tfp_direction_rewrite,
})
# dynamic_profit_max: stories that count shows or performances but price tickets leave revenue per unit of output
# undefined (the key treats the ticket price as the price of one unit of output).
TEMPLATE_FILTERS['dynamic_profit_max'] = lambda row: not re.search(r'\bticket', row['question_text'], re.I)

# Re-review: stories that call the production function a revenue, profit, quality, effectiveness or
# performance score (not a quantity of output sold at the stated price), and marginal / average cost stories that call
# the cost wL a cost per unit of output ("the cost for each imaging study is 6.45L"), are dropped.
_NOT_A_QUANTITY = re.compile(r'\b(benefit|revenue|profit|value|returns?|income|earnings|impact|effectiveness|score|'
                             r'satisfaction|performance(?!s)|quality)\w*', re.I)
_PER_OUTPUT_COST = re.compile(r'\b(each|per|every)\b(?![\w\s-]{0,30}\b(hour|labou?r|worker|employee|staff|volunteer|'
                              r'instructor|tutor|teacher|nurse|technician|crew|actor|coach)s?\b)[\w\s-]{0,40}$', re.I)


def _clause_before(text: str, placeholder: str) -> str:
    i = text.find('{' + placeholder + '}')
    return _SENTENCE_START.split(text[:i])[-1] if i >= 0 else ''


def _output_story_filter(p_name: str, cost: bool = False):
    def keep(row):
        text = row['question_text']
        if _NOT_A_QUANTITY.search(_clause_before(text, p_name)):
            return False
        return not (cost and _PER_OUTPUT_COST.search(_clause_before(text, 'c_function')))
    return keep


_dpm_tickets = TEMPLATE_FILTERS['dynamic_profit_max']
TEMPLATE_FILTERS.update({
    'dynamic_profit_max': lambda row: _dpm_tickets(row) and _output_story_filter('p_func')(row),
    'profit_maximization': _output_story_filter('p_func'),
    'marginal_cost': _output_story_filter('p_function', cost=True),
    'average_cost': _output_story_filter('p_function', cost=True),
})

_DPM_ASK = ('The production function gives the quantity of output, which sells at the stated price in each period. '
            'Answer with the increase in capital (K_2 minus K_1), not the new level of capital.')


_DPM_RATE_SUBS = [
    (re.compile(r'\b(discount(?:ed|s)?) (?:by|at) (?:a|the) rate (?:of |represented by )?\{discount\}', re.I),
     r'\1 by a factor of {discount}'),
    (re.compile(r'\bat a rate (?:of |represented by )?\{discount\}', re.I), 'by a factor of {discount}'),
    (re.compile(r'\bby (?:a|the) rate (?:of )?\{discount\}', re.I), 'by a factor of {discount}'),
    (re.compile(r'\bdiscount rate( of| represented by)? \{discount\}', re.I), r'discount factor\1 {discount}'),
    (re.compile(r'\b(?:at )?an? \{discount\} (?:discount )?(?:rate|factor)', re.I), 'a discount factor of {discount}'),
]


def _dpm_rewrite(row: pd.Series) -> Optional[str]:
    """dynamic_profit_max: 'the ideal capital investment amount for tomorrow' could mean K_2; the key is K_2 - K_1.
    The key multiplies tomorrow's revenue by {discount} (a discount factor); templates that call it a 'discount rate'
    or say 'discounted at a rate of' (as a rate, 1/(1 + r)) are reworded, and the convention is stated."""
    text = row['question_text']
    for pat, rep in _DPM_RATE_SUBS:
        text = pat.sub(rep, text)
    # any other wording ('discounted using the rate {discount}'): 'rate' -> 'factor' in that sentence
    text = ' '.join(re.sub(r'\brate\b', 'factor', sent, flags=re.I) if '{discount}' in sent else sent
                    for sent in re.split(r'(?<=[.!?])\s+', text))
    return (text.rstrip() + ' ' + _DPM_ASK + " Tomorrow's revenue is multiplied by the discount factor {discount}.")


TEMPLATE_REWRITES['dynamic_profit_max'] = _dpm_rewrite

# consumer-theory elements (review, October 2026): sampler/consumer_template_fixes.py
from autosteer.sampler import consumer_template_fixes as _consumer_fixes  # noqa: E402

_consumer_fixes.install(TEMPLATE_REWRITES)

# ----------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------

_LAST_Q = re.compile(r'[^.?!]*\?\s*$', re.S)


def _replace_last_question(text: str, new: str) -> Optional[str]:
    m = _LAST_Q.search(text.strip())
    if not m:
        return None
    return text.strip()[:m.start()].rstrip() + ' ' + new


_TRANSFER = ' after a suitable lump-sum redistribution of the initial holdings'


def _wt2_with_transfers(row: pd.Series) -> Optional[str]:
    # Second welfare theorem: a Pareto-efficient allocation is a competitive
    # equilibrium *after suitable lump-sum transfers*. Make the transfers explicit.
    text = row['question_text']
    if re.search(r'transfer|redistribut|lump', text, re.I):
        return text
    new, n = re.subn(r'\s*(?:,\s*)?(?:given|from|starting from|based on) (?:their|the|his|her|our|my|your) '
                     r'(?:initial|original|starting) [\w\s-]{1,40}?(?=\?|,|$)', _TRANSFER, text, count=1)
    if n:
        return new
    return re.sub(r'\?\s*$', ',' + _TRANSFER + '?', text.strip())


def _wt2_without_transfers(row: pd.Series) -> Optional[str]:
    return _replace_last_question(
        row['question_text'],
        'Without any redistribution of the initial holdings, is it guaranteed that some set of '
        'competitive equilibrium prices would lead them to this same final allocation?')


def _wt1_further_trades(row: pd.Series) -> Optional[str]:
    # a few originals already ask "is this allocation Pareto efficient?" (answer
    # yes); drop them from the 'no' variant (the pareto variant covers them)
    m = _LAST_Q.search(row['question_text'].strip())
    last = m.group(0).lower() if m else ''
    if re.search(r'pareto[- ](efficient|optimal)', last):
        return None
    return row['question_text']


def _wt1_pareto(row: pd.Series) -> Optional[str]:
    return _replace_last_question(
        row['question_text'],
        'According to the first welfare theorem, is the final allocation Pareto efficient?')


# element -> [(type, rewrite)]: every template is emitted once per variant
# with that type (the first variant keeps the original question id). The
# welfare-theorem fillers key the answer on the type, so both 'yes' and
# 'no' occur (the published answers were constant). Narun's default.
TEMPLATE_VARIANTS: dict[str, list[tuple[str, Callable[[pd.Series], Optional[str]]]]] = {
    'welfare_theorem_1': [('further_trades', _wt1_further_trades),                # -> no
                          ('pareto_efficient', _wt1_pareto)],                     # -> yes
    'welfare_theorem_2': [('with_transfers', _wt2_with_transfers),                # -> yes
                          ('without_transfers', _wt2_without_transfers)],         # -> no
}


def _apply_variants(element, q, opts):
    variants = TEMPLATE_VARIANTS.get(element)
    if not variants or q is None or len(q) == 0:
        return q, opts
    next_id = int(q['question_id'].max()) + 1
    ids = sorted(q['question_id'].unique())
    new_q, new_o = [], []
    for k, (typ, fn) in enumerate(variants):
        clone = q.copy()
        clone['question_text'] = [fn(r) for _, r in clone.iterrows()]
        clone['type'] = typ
        if k:
            id_map = {old: next_id + i for i, old in enumerate(ids)}
            next_id += len(ids)
            clone['question_id'] = clone['question_id'].map(id_map)
            if opts is not None and len(opts):
                o = opts[opts['question_id'].isin(ids)].copy()
                o['question_id'] = o['question_id'].map(id_map)
                new_o.append(o)
        elif opts is not None and len(opts):
            new_o.append(opts)
        clone = clone[clone['question_text'].notna()]
        new_q.append(clone)
    q = pd.concat(new_q, ignore_index=True)
    if new_o:
        opts = pd.concat(new_o, ignore_index=True)
    return q, opts


# ----------------------------------------------------------------------
# Difficulty
# ----------------------------------------------------------------------

# Published difficulty is 0/1/2. Some elements use their own internal levels,
# which the fillers branch on; they are mapped only in the sampler output
# (and in SamplerConfig filtering / the manifest). Narun's default, flagged.
DIFFICULTY_MAPS: dict[str, dict[int, int]] = {
    # 0: 2 goods, prices given; 1: 2 goods, functional form; 3: 3 goods,
    # linear utility, prices given; 4: 3 goods, prices given;
    # 2: 3 goods, functional form
    'deriving_hicksian_demand': {0: 0, 1: 1, 3: 1, 4: 1, 2: 2},
    # 5: single prospect; 6: scenarios with a population / several options
    'expected_value_counts_aug2024': {5: 0, 6: 1},
}


def public_difficulty(element: str, level) -> int:
    level = int(level)
    return DIFFICULTY_MAPS.get(element, {}).get(level, level)


# ----------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------


def _fix_frame(element: str, q: pd.DataFrame, opts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if q is None or len(q) == 0:
        return q, opts
    q = q.copy()
    opts = opts.copy() if opts is not None else opts
    index_cols = [c for c in ('question_id', 'sub_id') if c in q.index.names]
    if index_cols:
        q = q.reset_index()
    if opts is not None and any(n in ('question_id', 'sub_id') for n in (opts.index.names or [])):
        opts = opts.reset_index()

    renames = TYPE_RENAMES.get(element)
    if renames:
        q['type'] = q['type'].replace(renames)

    expansions = TYPE_EXPANSIONS.get(element)
    if expansions:
        next_id = int(q['question_id'].max()) + 1
        new_q, new_o = [q], [opts]
        for src, targets in expansions.items():
            base = q[q['type'].str.contains(src, regex=False)]
            for tgt in targets:
                ids = sorted(base['question_id'].unique())
                id_map = {old: next_id + i for i, old in enumerate(ids)}
                next_id += len(ids)
                clone = base.copy()
                clone['type'] = clone['type'].str.replace(src, tgt, regex=False)
                clone['question_id'] = clone['question_id'].map(id_map)
                new_q.append(clone)
                if opts is not None and len(opts):
                    o = opts[opts['question_id'].isin(ids)].copy()
                    o['question_id'] = o['question_id'].map(id_map)
                    new_o.append(o)
        q = pd.concat(new_q, ignore_index=True)
        if opts is not None:
            opts = pd.concat(new_o, ignore_index=True)

    q, opts = _apply_variants(element, q, opts)

    keep_filter = TEMPLATE_FILTERS.get(element)
    rewrite = TEMPLATE_REWRITES.get(element)
    if keep_filter or rewrite:
        texts, keep = [], []
        for _, row in q.iterrows():
            ok = keep_filter(row) if keep_filter else True
            text = row['question_text']
            if ok and rewrite:
                text = rewrite(row)
                ok = text is not None
            keep.append(ok)
            texts.append(text if text is not None else row['question_text'])
        q['question_text'] = texts
        dropped = set(q.loc[[not k for k in keep], 'question_id'])
        # a multi-part question is dropped as a whole
        q = q[~q['question_id'].isin(dropped)].reset_index(drop=True)
        if opts is not None and len(opts):
            opts = opts[~opts['question_id'].isin(dropped)].reset_index(drop=True)
    return q, opts


def has_fixes(element: str) -> bool:
    return any(element in d for d in (TYPE_EXPANSIONS, TYPE_RENAMES, TEMPLATE_FILTERS, TEMPLATE_REWRITES,
                                      TEMPLATE_VARIANTS))


def _placeholder_names(frames) -> set[str]:
    import string
    names = set()
    for q in frames:
        if q is None or len(q) == 0:
            continue
        for text in q['question_text']:
            try:
                names |= {f for _, f, _, _ in string.Formatter().parse(text) if f}
            except ValueError:
                continue
    return names


def _drop_bare_placeholders(q: pd.DataFrame, opts: pd.DataFrame, names: set[str]):
    """
    Drop templates that spell a placeholder name without braces (e.g.
    "there are num_victims individuals"); they would be shown unfilled.
    Only names containing an underscore are checked, to avoid plain words.
    """
    names = sorted(n for n in names if '_' in n)
    if q is None or len(q) == 0 or not names:
        return q, opts
    pat = re.compile(r'(?<![{\w])(' + '|'.join(map(re.escape, names)) + r')(?![}\w])')
    q = q.copy()
    if 'question_id' not in q.columns:
        q = q.reset_index()
    bad = set(q.loc[q['question_text'].apply(lambda t: bool(pat.search(t))), 'question_id'])
    if not bad:
        return q, opts
    q = q[~q['question_id'].isin(bad)].reset_index(drop=True)
    if opts is not None and len(opts):
        if 'question_id' not in opts.columns:
            opts = opts.reset_index()
        opts = opts[~opts['question_id'].isin(bad)].reset_index(drop=True)
    return q, opts


# Element-specific selection hooks, applied last: fn(element, questions, options) -> (questions, options).
def _multi_agent_hook(element, q, opts):
    from autosteer.filler_tools.multi_agent_tools import select_multi_agent_templates
    return select_multi_agent_templates(element, q, opts)


# constrained_optimization: the data directory's templates are no longer used (code templates in
# filler_tools/foundations_tools.py, October 2026), so its template filter was removed.


TEMPLATE_HOOKS: dict[str, Callable] = {
    **{e: _multi_agent_hook for e in ('trigger', 'enforceability', 'backward_induction')},
}
# completeness, completeness_risk (code templates, completeness_tools.py) and feasibility (code templates,
# feasibility_tools.py) no longer use the data directory's templates; their hooks were removed (October 2026).


def exact_copies(q, o) -> set:
    """question_ids of the templates that repeat an earlier one exactly: every part's text, the template's options and
    its metadata. The data has none (autosteer.data_cleanup deletes them), but the corrections can make some:
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


def apply_template_fixes(element: str, dataset, hooks: bool = True):
    """Apply the corrections for `element` to `dataset` in place and return it."""
    if has_fixes(element):
        dataset.questions, dataset.options = _fix_frame(element, dataset.questions, dataset.options)
        dataset.example_questions, dataset.example_options = _fix_frame(
            element, dataset.example_questions, dataset.example_options
        )
    # '{{price}}' is an escaped placeholder: str.format prints it as a literal
    # '{price}'. Unescape it so it is filled, or dropped by the sampler's
    # fillability check if the filler does not provide it.
    for attr in ('questions', 'example_questions'):
        q = getattr(dataset, attr)
        if q is not None and len(q):
            q = q.copy()
            q['question_text'] = q['question_text'].str.replace(r'\{\{(\w+)\}\}', r'{\1}', regex=True)
            setattr(dataset, attr, q)
    names = _placeholder_names([dataset.questions, dataset.example_questions])
    dataset.questions, dataset.options = _drop_bare_placeholders(dataset.questions, dataset.options, names)
    dataset.example_questions, dataset.example_options = _drop_bare_placeholders(
        dataset.example_questions, dataset.example_options, names
    )
    hook = TEMPLATE_HOOKS.get(element) if hooks else None
    if hook:
        for q_attr, o_attr in (('questions', 'options'), ('example_questions', 'example_options')):
            q, o = getattr(dataset, q_attr), getattr(dataset, o_attr)
            if q is None or len(q) == 0:
                continue
            if 'question_id' not in q.columns:
                q = q.reset_index()
            if o is not None and len(o) and 'question_id' not in o.columns:
                o = o.reset_index()
            q, o = hook(element, q, o)
            setattr(dataset, q_attr, q)
            setattr(dataset, o_attr, o)
    for q_attr, o_attr in (('questions', 'options'), ('example_questions', 'example_options')):
        q, o = getattr(dataset, q_attr), getattr(dataset, o_attr)
        copies = exact_copies(q, o)
        if copies:
            q = q.reset_index() if 'question_id' not in q.columns else q
            setattr(dataset, q_attr, q[~q['question_id'].isin(copies)].reset_index(drop=True))
            if o is not None and len(o):
                o = o.reset_index() if 'question_id' not in o.columns else o
                setattr(dataset, o_attr, o[~o['question_id'].isin(copies)].reset_index(drop=True))
    return dataset


# ----------------------------------------------------------------------
# Review (October 2026): conventions and story filters for arbitrage, aggregation_of_labor_supply and
# deriving_labor_supply (pipeline templates).
# ----------------------------------------------------------------------

def _before_question(text: str, sentence: str) -> str:
    """`sentence` inserted before the last sentence of `text` (its question)."""
    text = text.rstrip()
    starts = [m.end() for m in _SENTENCE_START.finditer(text)]
    if not starts:
        return text + ' ' + sentence
    cut = starts[-1]
    return text[:cut] + sentence + ' ' + text[cut:]


ARBITRAGE_CONVENTION = ('All prices and costs are in US dollars, and the same transport cost applies to shipping in '
                        'either direction.')


def _arbitrage_rewrite(row: pd.Series) -> Optional[str]:
    """The reverse direction's transport cost and the currency were not stated (the "No" key relies on both)."""
    return _before_question(row['question_text'], ARBITRAGE_CONVENTION)


AGG_LABOR_SUPPLY_CONVENTION = ('Here P is the hourly wage in dollars, each function gives the hours of labor supplied '
                               'per week to a single member of its group, and no worker supplies labor to more than '
                               'one of them.')
# stories in which the "labor supply function" is a hiring or policy strategy, not workers' supply
_AGG_LS_STORY = re.compile(r'strateg|recruit|committee|council|think[- ]tank|legislat|campaign|mandat|politic', re.I)
_HOW_QUESTION = re.compile(r'^\s*(how\b|what (?:process|steps?|method|approach))', re.I)


def _agg_labor_supply_filter(row: pd.Series) -> bool:
    return not _AGG_LS_STORY.search(row['question_text'])


def _agg_labor_supply_rewrite(row: pd.Series) -> Optional[str]:
    text = _segment_count_rewrite(row)
    if text is None:
        return None
    if row['type'] == 'supply_function':
        last = _LAST_Q.search(text)
        if last and _HOW_QUESTION.match(last.group(0)):
            text = text[:last.start()].rstrip() + ' What is the aggregate labor supply function?'
        return _before_question(text, AGG_LABOR_SUPPLY_CONVENTION)
    return _before_question(text, AGG_LABOR_SUPPLY_CONVENTION) + ' Round to two decimals.'


# income that is not an hourly wage over a 24-hour day (weekly / monthly hours, salaries, piece rates, match earnings,
# practice hours that are not paid work)
_DLS_STORY = re.compile(r'week|month|salar|annual|\byears?\b|per (?:item|piece|unit|product|session|lesson|class|gig|'
                        r'show|performance|project|article|game)|price|\bp\b|match|tournament|practi[cs]|commission|'
                        r'\bfee', re.I)


def _deriving_labor_supply_filter(row: pd.Series) -> bool:
    return not _DLS_STORY.search(row['question_text'])


TEMPLATE_FILTERS.update({
    'aggregation_of_labor_supply': _agg_labor_supply_filter,
    'deriving_labor_supply': _deriving_labor_supply_filter,
})
TEMPLATE_REWRITES.update({
    'arbitrage': _arbitrage_rewrite,
    'aggregation_of_labor_supply': _agg_labor_supply_rewrite,
})
# Independent-review fixes for behavioural elements (October 2026): sampler/behav_fixes.py
from . import behav_fixes as _behav  # noqa: E402

TEMPLATE_FILTERS.update(_behav.FILTERS)
TEMPLATE_REWRITES.update(_behav.REWRITES)
TEMPLATE_VARIANTS.update(_behav.VARIANTS)
TEMPLATE_HOOKS.update(_behav.HOOKS)
for _e, _after in _behav.AFTER_HOOKS.items():
    TEMPLATE_HOOKS[_e] = (lambda first, second: lambda e, q, o: second(e, *first(e, q, o)))(TEMPLATE_HOOKS[_e], _after)

import re
# Built-in packages
import random
from fractions import Fraction
from string import ascii_uppercase
ALPHABET = list(ascii_uppercase)
# External packages
import numpy as np
# Local packages
from .fill_registry import FillRegister
function_registry = FillRegister()
# from .fill_registry import function_registry

class Gamble:
    def __init__(self, num_outcomes=2, payoff_type=int, payoff_range=(0, 100), risk_free_premium=None):
        self.num_outcomes = num_outcomes
        self.payoff_type = payoff_type
        self.payoff_range = payoff_range
        self.risk_free_premium = risk_free_premium
        self.payoffs = [self._generate_payoff() for _ in range(num_outcomes)]
        if self.risk_free_premium is not None:
            self.payoffs[-1] = self.risk_free_premium
        self.probabilities = self._generate_probabilites()
    
    def _generate_payoff(self):
        if self.payoff_type == int:
            assert all(isinstance(elm, int) for elm in self.payoff_range), "Non-integer payoff range provided for integer payoff type."
            return random.randint(*self.payoff_range)
        elif self.payoff_type == float:
            return round(random.uniform(0, 1) * self.payoff_range[1] + self.payoff_range[0], 2)
    
    # NOTE: may care about having more complex distributions that depend on the index
    def _generate_probability(self, outcome_index):
        return random.uniform(0, 1)
    
    def _generate_probabilites(self):
        probabilities = [self._generate_probability(i) for i in range(self.num_outcomes)]
        probabilities = [prob/sum(probabilities) for prob in probabilities]
        return probabilities
    
    @classmethod
    def from_payoffs_and_probabilities(cls, payoffs, probabilities):
        gamble = cls(num_outcomes=len(payoffs))
        gamble.payoffs = payoffs
        gamble.probabilities = probabilities
        return gamble

    @classmethod
    def from_payoffs(cls, payoffs):
        gamble = cls(num_outcomes=len(payoffs))
        gamble.payoffs = payoffs
        return gamble
    
    @classmethod
    def from_probabilities(cls, probabilities, payoff_type=int, payoff_range=(0, 100), risk_free_premium=None):
        gamble = cls(num_outcomes=len(probabilities), payoff_type=payoff_type, payoff_range=payoff_range, risk_free_premium=risk_free_premium)
        gamble.probabilities = probabilities
        return gamble

    def compute_expectation(self):
        expectation = sum(payoff * probability for payoff, probability in zip(self.payoffs, self.probabilities))
        return expectation
    
    def compute_riskiness(self):
        expectation = self.compute_expectation()
        standard_deviation = np.sqrt(sum((payoff - expectation) ** 2 * probability for payoff, probability in zip(self.payoffs, self.probabilities)))
        return standard_deviation
    
    def to_dict(self):
        return {
            'payoffs': [f"{payoff:.2f}" if self.payoff_type == float else f"{payoff}" for payoff in self.payoffs],
            'probabilities': [f"{prob*100:.2f}%" for prob in self.probabilities]
        }
    
class Scenario:
    def __init__(self, scenario, payoff_range=(0, 1000), payoff_type=int, risk_free_premium = None):
        assert type(scenario) == dict, 'Scenario must be a dictionary.'
        assert scenario['scenario_type'] in ['certainty', 'reflection', 'risk_aversion', 'risk_seeking'], 'Invalid scenario type, must be one of certainty, reflection, risk_aversion, risk_seeking'
        self.scenario_type = scenario['scenario_type']
        if self.scenario_type in ['risk_aversion', 'risk_seeking']:
            assert 'gamble_offset' in scenario, "Gamble offset ('gamble_offset') must be provided for risk_aversion and risk_seeking scenarios."
            if self.payoff_type == int:
                assert type(scenario['gamble_offset']) == int, "Gamble offset ('gamble_offset') must be an integer."
            assert scenario['gamble_offset'] > 0, "Gamble offset ('gamble_offset') must be greater than 0."
        self.scenario = scenario
        self.payoff_range = payoff_range
        self.payoff_type = payoff_type
        self.risk_free_premium = risk_free_premium
        self.scenarios = self._generate_scenario()
    
    def _generate_scenario(self):
        if self.scenario_type == 'certainty':
            scenarios = self.gen_certainty_scenario()
            return scenarios
        elif self.scenario_type == 'reflection':
            self.payoff_range = (-self.payoff_range[1], -self.payoff_range[0])
            scenarios = self.gen_certainty_scenario()
            return scenarios
        elif self.scenario_type == 'risk_aversion':
            scenarios = (
                Gamble(num_outcomes=2, payoff_type=self.payoff_type, payoff_range=self.payoff_range),
                Gamble(num_outcomes=2, payoff_type=self.payoff_type, payoff_range=self.payoff_range)
            )
            # NOTE: this is to ensure that the first scenario is riskier and has a higher expectation (i.e., the first option is correct)
            while scenarios[1].compute_expectation() + self.scenario['gamble_offset'] != scenarios[0].compute_expectation() or scenarios[1].compute_riskiness() > scenarios[0].compute_riskiness():
                scenarios = (
                    Gamble(num_outcomes=2, payoff_type=self.payoff_type, payoff_range=self.payoff_range),
                    Gamble(num_outcomes=2, payoff_type=self.payoff_type, payoff_range=self.payoff_range)
                )
            return (scenarios,)
        elif self.scenario_type == 'risk_seeking':
            scenarios = (
                Gamble(num_outcomes=2, payoff_type=self.payoff_type, payoff_range=self.payoff_range),
                Gamble(num_outcomes=2, payoff_type=self.payoff_type, payoff_range=self.payoff_range)
            )
            # NOTE: this is to ensure that the second scenario is riskier and has a lower expectation (i.e., the first option is correct)
            while scenarios[1].compute_expectation() + self.scenario['gamble_offset'] != scenarios[0].compute_expectation() or scenarios[1].compute_riskiness() < scenarios[0].compute_riskiness():
                scenarios = (
                    Gamble(num_outcomes=2, payoff_type=self.payoff_type, payoff_range=self.payoff_range),
                    Gamble(num_outcomes=2, payoff_type=self.payoff_type, payoff_range=self.payoff_range)
                )
            return (scenarios,)
    
    def gen_certainty_scenario(self):
        '''
        Generates certainty effect question with two scenarios, with gambles of two outcomes and a risk-free premium of 0
        '''
        # Scenario 1
        gamble_a = Gamble(num_outcomes=1, payoff_type=self.payoff_type, payoff_range=self.payoff_range) 
        random.randint(*self.payoff_range) if self.payoff_type == int else round(random.uniform(0, 1) * self.payoff_range[1] + self.payoff_range[0], 2)
        gamble_b = Gamble(payoff_type = self.payoff_type, payoff_range = self.payoff_range, risk_free_premium=0)
        # this is to ensure that there cannot exist a utility function that makes sense given this particular setup
        while gamble_b.compute_expectation() < gamble_a.compute_expectation() and gamble_b.probabilities[0] < (1-gamble_b.probabilities[0])/gamble_b.probabilities[0]:
            gamble_b = Gamble(payoff_type = self.payoff_type, payoff_range = self.payoff_range, risk_free_premium=0)
        
        # Scenario 2
        gamble_c = Gamble.from_payoffs(payoffs=[gamble_a.payoffs[0], 0])
        gamble_d = Gamble.from_probabilities(probabilities=[1 - elm for elm in gamble_b.probabilities], risk_free_premium=0)
        while gamble_d.compute_expectation() < gamble_c.compute_expectation():
            gamble_c = Gamble.from_payoffs(payoffs=[gamble_a.payoffs[0], 0])
            gamble_d = Gamble.from_probabilities(probabilities=[1 - elm for elm in gamble_b.probabilities], risk_free_premium=0)

        # NOTE: choosing gamble_a and gamble_d is the inconsistent choice
        return (
            (gamble_a, gamble_b),
            (gamble_c, gamble_d)
        )
# completeness, completeness_risk: filler_tools/completeness_tools.py (code templates, October 2026)

def _eu_options(answer, spread, fmt, nonneg=True):
    """The answer plus three distractors at least 5% (relative) apart, all rendered with fmt."""
    vals = [answer]
    while len(vals) < 4:
        cand = answer + random.choice([-1, 1]) * random.uniform(0.1, 1.0) * spread
        if nonneg and cand < 0:
            continue
        if all(abs(cand - v) > max(0.05 * abs(v), 0.02) for v in vals):
            vals.append(cand)
    texts = [fmt(v) for v in vals]
    correct = texts[0]
    random.shuffle(texts)
    return [texts], [[t == correct for t in texts]]


_EU_BAD = re.compile(
    r"confin\w*|indoors?|inside|snowstorm\w*|snowed|heavy snow\w*|snowfall|storm\w*|blizzard|rain\w*|fog|bad weather|poor weather|stuck|"
    r"unable|cannot|can't|miss\w*|lost|los[et]\b|not (?:be )?(?:saved|cured|surviv\w*|recover\w*)|"
    r"die|dies|died|death|dead|fatal\w*|harm\w*|side effects?|complications?|adverse|fail\w*|succumb\w*")


def _eu_ask_clause(text):
    """The interrogative clause of the final sentence ('how many days ... skiing?')."""
    sentences = [x for x in re.split(r'(?<=[.!?])\s+', text.strip()) if x]
    last = sentences[-1].lower()
    wh = list(re.finditer(r"\b(how many|how much|what|which|calculate|determine|compute|estimate|find)\b", last))
    return last[wh[-1].start():] if wh else last


def _eu_asks_bad(metadata):
    """
    From the template text (passed by the sampler as metadata['question_text']),
    does the final question ask for the bad outcome (bad days, patients not
    saved, money lost)? Falls back to the tags when no text is given.
    Phrases like "without heavy snowfall" / "without side effects" count as
    asking for the good outcome.
    """
    text = metadata.get('question_text')
    if not text:
        tags = metadata.get('tags', '')
        return any(t in tags.split(';') for t in ('bad', 'lose', 'not_save'))
    last = _eu_ask_clause(text)
    # "without skiing", "off the slopes": the good activity is missed -> bad
    if re.search(r"\bwithout (?:\w+ )?(?:ski|hik|go\b|going|explor|enjoy|surf|play)|off the slopes|not (?:be able to )?(?:ski|hike|go)", last):
        return True
    # "without (harmful) side effects": the bad outcome is excluded -> good
    last = re.sub(r"\bwithout [^,.?;]*", " ", last)
    return bool(_EU_BAD.search(last))


@function_registry.register('expected_value_counts_aug2024')
def gen_compute_expected_utility(metadata):
    """
    Expected values of two-outcome prospects (risk neutral), keyed by the
    template tags, from the displayed (2-decimal percent) probabilities:
      medical (tags 'save' / other): N p_save or N (1 - p_save)
      vacation_planning ('good' / 'bad'): n p_good or n p_bad
      gambling ('win' / 'lose', 'multiple'): NET value per game
        EV = p_win W - p_lose L (convention: net when a cost is stated),
        answer n EV for 'win' and the expected loss -n EV for 'lose'.
    """
    pct = lambda: round(random.uniform(0.05, 0.95) * 100, 2)
    tags = metadata['tags']
    if metadata['domain'] == 'medical':
        num_patients = random.randint(10, 100)
        save = pct()
        not_save = round(100 - save, 2)
        p = not_save if _eu_asks_bad(metadata) else save
        answer = num_patients * p / 100
        options, mask = _eu_options(answer, max(answer, 1), lambda v: f"{v:.2f}")
        return {
            'num_patients': num_patients,
            'save_chance': f"{save:.2f}%",
            'not_save_chance': f"{not_save:.2f}%",
            'options_list': options,
            'correct_answer': mask,
        }
    elif metadata['domain'] == 'gambling':
        win = pct()
        lose = round(100 - win, 2)
        win_amount = random.randint(1, 100)
        lose_amount = random.randint(1, 100)
        num_games = random.randint(2, 100) if 'multiple' in tags else 1
        # the loss only counts if the template states it
        text_ = metadata.get('question_text')
        cost = lose_amount if (text_ is None or '{lose_amount}' in text_) else 0
        ev = win / 100 * win_amount - lose / 100 * cost
        text = (metadata.get('question_text') or '').lower()
        last = _eu_ask_clause(text) if text else ''
        signed = re.search(r"(win|won|gain)\w* \(?or (lose|lost|loss)|value of (the|this) game", last)
        asks_loss = (bool(re.search(r"\blos[et]|\bloss|\bcost", last)) and not signed) if text else 'lose' in tags
        answer = -num_games * ev if asks_loss else num_games * ev
        options, mask = _eu_options(answer, max(abs(answer), num_games), lambda v: f"${v:.2f}", nonneg=False)
        return {
            'win_chance': f"{win:.2f}%",
            'lose_chance': f"{lose:.2f}%",
            'win_amount': f"${win_amount}",
            'lose_amount': f"${lose_amount}",
            'num_games': num_games,
            'options_list': options,
            'correct_answer': mask,
        }
    elif metadata['domain'] == 'vacation_planning':
        good = pct()
        bad = round(100 - good, 2)
        num_days = random.randint(2, 10)
        p = bad if _eu_asks_bad(metadata) else good
        answer = num_days * p / 100
        options, mask = _eu_options(answer, max(answer, 1), lambda v: f"{v:.2f}")
        return {
            'good_chance': f"{good:.2f}%",
            'bad_chance': f"{bad:.2f}%",
            'num_days': num_days,
            'options_list': options,
            'correct_answer': mask,
        }
    raise ValueError(f"Unknown compute_expected_utility domain {metadata['domain']!r}")

# --------------------------------------------------------------------------- certainty effect
# Allais / common-ratio pairs (Kahneman & Tversky). One item = two question parts in the same story:
#   part 0:  A = X for sure                    vs  B = Y with probability p (else nothing)
#   part 1:  C = X with probability k          vs  D = Y with probability k*p (else nothing)
# Expected utility (u increasing, u(0) = 0) implies A > B  <=>  C > D, so A&C and B&D are consistent,
# A&D is the certainty effect and B&C the reverse violation. Templates (data/templates/certainty_effect,
# built by build_templates.py there) list the sure-side option (A / C) first; the release exporter
# (src/release/export_certainty_effect.py) shuffles and records which option is the sure side.
CE_P_PCT = (60, 95)                   # p, integer percent
CE_K_PCT = tuple(range(10, 55, 5))    # k in {10%, 15%, ..., 50%}: k*p has at most 2 decimals in percent
CE_EV_TOL = Fraction(1, 20)           # |pY - X| >= 5% of X, so a risk-neutral agent is never (nearly) indifferent
CE_EV_RATIO = {'lottery': (1.05, 1.40), 'sure': (0.70, 0.95)}   # pY / X, by the side a risk-neutral agent prefers
CE_SCALES = {
    # (step, lo, hi): X = step * randint(lo, hi); Y is rounded to the same step
    'money': [(10, 10, 99), (50, 4, 99), (100, 10, 99), (500, 4, 40)],
    'count': [(1, 20, 99), (5, 10, 99), (10, 10, 90)],
}


def fmt_pct(q) -> str:
    """Exact percent string of a percent value with at most 2 decimals (trailing zeros dropped): 20, 12.45, 22.5."""
    hundredths = Fraction(q) * 100
    assert hundredths.denominator == 1 and hundredths >= 0, q
    whole, cents = divmod(int(hundredths), 100)
    return (f"{whole}" if cents == 0 else f"{whole}.{cents:02d}".rstrip('0')) + '%'


def indefinite_article(pct: str) -> str:
    """'a' or 'an' before a spoken percent: an 8%, an 11.5%, an 18%, an 80-89%; a 60%, a 7.2%."""
    whole = int(pct.rstrip('%').split('.')[0])
    return 'an' if whole in (8, 11, 18) or 80 <= whole <= 89 else 'a'


def draw_common_ratio(kind: str = 'money', favor: str = None) -> dict:
    """
    Numbers for one common-ratio pair. Returns ints X < Y, p_pct, k_pct and the exact k*p (Fraction, percent).
    `favor` is the side a risk-neutral agent prefers ('lottery': pY > X, 'sure': pY < X); drawn 50/50 if None.
    """
    favor = favor or random.choice(['lottery', 'sure'])
    lo_r, hi_r = CE_EV_RATIO[favor]
    for _ in range(10000):
        step, lo, hi = random.choice(CE_SCALES[kind])
        X = step * random.randint(lo, hi)
        p = random.randint(*CE_P_PCT)
        k = random.choice(CE_K_PCT)
        Y = step * round(random.uniform(lo_r, hi_r) * X * 100 / p / step)
        r = Fraction(p * Y, 100 * X)            # exact EV ratio pY / X on the displayed numbers
        if Y <= X or Y < X * Fraction(21, 20):  # Y clearly larger than X
            continue
        if abs(r - 1) < CE_EV_TOL or not (lo_r <= r <= hi_r):
            continue
        return {'X': X, 'Y': Y, 'p_pct': p, 'k_pct': k, 'kp_pct': Fraction(k * p, 100), 'favor': favor}
    raise RuntimeError('could not draw a common-ratio pair')


@function_registry.register('certainty_effect')
def gen_certainty_effect(metadata):
    """Placeholder values for a certainty_effect template (counts for domain 'medical', else money); *_art = 'a'/'an'."""
    d = draw_common_ratio('count' if metadata.get('domain') == 'medical' else 'money')
    out = {
        'x_amount': f"{d['X']:,}",
        'y_amount': f"{d['Y']:,}",
        'p_pct': fmt_pct(d['p_pct']),
        'k_pct': fmt_pct(d['k_pct']),
        'kp_pct': fmt_pct(d['kp_pct']),
    }
    for key in ('p', 'k', 'kp'):
        out[f'{key}_art'] = indefinite_article(out[f'{key}_pct'])
    return out


# --------------------------------------------------------------------------- reflection effect
# Gain / loss mirror pairs (Kahneman & Tversky 1984; STEER Illustration 2.5.c), same story in both parts:
#   part 0 (gains):   sure +X   vs  +Y with probability p (else nothing)
#   part 1 (losses):  sure -X   vs  -Y with probability p (else nothing)
# with EXACTLY equal expected values p*Y = X on the displayed numbers. Any expected-utility maximiser whose
# risk attitude does not change sign between the gain and loss range (u concave throughout, or convex
# throughout) picks the same side in both parts; sure-gain & risky-loss is the reflection effect. With unequal
# EVs a risk-neutral agent would itself "flip" (EV favours one side for gains and the other for losses), so
# the EV must be equal. Templates list the sure side first; see src/release/consistency.py (reflection_same_side).
RE_P_PCT = tuple(range(20, 85, 5))   # p in {20%, 25%, ..., 80%}
RE_SCALES = {
    # (step, lo, hi): Y = step * randint(lo, hi); step is a multiple of 20 so X = p*Y/100 is an integer
    'money': [(20, 10, 99), (100, 10, 99), (500, 4, 40)],
    'count': [(20, 5, 45)],
}


def draw_reflection(kind: str = 'money') -> dict:
    """Ints X < Y and p_pct with p*Y = X exactly (equal expected value)."""
    step, lo, hi = random.choice(RE_SCALES[kind])
    p = random.choice(RE_P_PCT)
    Y = step * random.randint(lo, hi)
    X = Fraction(p * Y, 100)
    assert X.denominator == 1 and 0 < X < Y
    return {'X': int(X), 'Y': Y, 'p_pct': p}


@function_registry.register('reflection_effect')
def gen_reflection_effect(metadata):
    """Placeholder values for a reflection_effect template (counts for domain 'medical', else money); amounts are
    unsigned, the template wording carries the gain / loss frame."""
    d = draw_reflection('count' if metadata.get('domain') == 'medical' else 'money')
    p = fmt_pct(d['p_pct'])
    return {'x_amount': f"{d['X']:,}", 'y_amount': f"{d['Y']:,}", 'p_pct': p, 'p_art': indefinite_article(p)}


@function_registry.register('risk_aversion')
def gen_risk_aversion(metadata):
    if metadata['domain'] == 'medical':
        payoff_range = (5, 50)
    elif metadata['domain'] == 'finance':
        if metadata['type'] == 'percent':
            payoff_range = (0.01, 0.5)
        else:
            payoff_range = (5, 50)
    offset = random.randint(1, 100)
    scenarios = Scenario({'scenario_type': 'risk_aversion', 'offset': offset}, payoff_range=payoff_range)
    gamble_a, gamble_b = scenarios.scenarios[0]
    return {
        'return_a_one': gamble_a.to_dict()['payoffs'][0],
        'prob_a_one': gamble_a.to_dict()['probabilities'][0],
        'return_a_two': gamble_a.to_dict()['payoffs'][1],
        'prob_a_two': gamble_a.to_dict()['probabilities'][1],
        'return_b_one': gamble_b.to_dict()['payoffs'][0],
        'prob_b_one': gamble_b.to_dict()['probabilities'][0],
        'return_b_two': gamble_b.to_dict()['payoffs'][1],
        'prob_b_two': gamble_b.to_dict()['probabilities'][1],
    }

@function_registry.register('risk_seeking')
def gen_risk_seeking(metadata):
    if metadata['domain'] == 'medical':
        payoff_range = (5, 50)
    elif metadata['domain'] == 'finance':
        if metadata['type'] == 'percent':
            payoff_range = (0.01, 0.5)
        else:
            payoff_range = (5, 50)
    offset = random.randint(1, 100)
    scenarios = Scenario({'scenario_type': 'risk_seeking', 'offset': offset}, payoff_range=payoff_range)
    gamble_a, gamble_b = scenarios.scenarios[0]
    return {
        'return_a_one': gamble_a.to_dict()['payoffs'][0],
        'prob_a_one': gamble_a.to_dict()['probabilities'][0],
        'return_a_two': gamble_a.to_dict()['payoffs'][1],
        'prob_a_two': gamble_a.to_dict()['probabilities'][1],
        'return_b_one': gamble_b.to_dict()['payoffs'][0],
        'prob_b_one': gamble_b.to_dict()['probabilities'][0],
        'return_b_two': gamble_b.to_dict()['payoffs'][1],
        'prob_b_two': gamble_b.to_dict()['probabilities'][1],
    }

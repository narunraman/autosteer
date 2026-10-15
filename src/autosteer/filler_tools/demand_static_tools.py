# Built-in packages
from string import ascii_uppercase
# External packages
import numpy as np
# Local packages
from .utilities.micro_utility_utils import UtilityFunction
from .utilities.filler_utils import rng
from .utilities.rounding import near_tie
from .fill_registry import FillRegister
from .utilities.filler_utils import gen_answer_and_options
from autosteer.filler_tools.rng import legacy_np, pyrng

function_registry = FillRegister()
ALPHABET = list(ascii_uppercase)

class HicksianDemandFunction:
    # TODO: superlcass DemandFunction
    # NOTE: must call from_utility_fn to derive the demand function
    
    def __init__(self, kind: str):
        assert kind in ['linear', 'quadratic', 'cobb-douglas'] # TODO: currently only support hicksian demand;
        self.kind = kind
        # Generate random coefficients for the demand function
        
        if kind == 'linear':
            self.coefficients = [rng.uniform(-10, -1), rng.uniform(1, 10)]  # Ensure the slope is negative
        else:  # Quadratic
            self.coefficients = [rng.uniform(-10, -1), rng.uniform(-10, -1), rng.uniform(1, 10)]  # Ensure the quadratic term is negative
        

    @classmethod
    def from_utility_fn(cls, utility_function: UtilityFunction, utility_level: float):
        """
        Initializes the hicksian demand function (from an utility function).

        Parameters:
        - utility_function: An instance of the UtilityFunction class.
        - utility_level: the level of utility consumer wants to reach.
        """
        assert utility_function.family in ['cobb-douglas', 'linear'], "Only Cobb-Douglas and Linear" 
        # Create a new instance of the Hicksian demand class
        instance = cls(utility_function.family)
        # Set the utility function and utility_level of the instance
        instance.utility_function = utility_function
        instance.utility_level = utility_level
        return instance

    def _cobb_douglas_utility(self, prices):
        # 3 GOODS MAX
        # DIFFICULTY LEVELS: 3 different levels (basis on prices given)
        """
        Computes the hicksian demand for goods given Cobb-Douglas utility function and prices.

        Parameters:
        - prices: A list of prices for each good.

        Returns:
        - demand: A list of (adjusted) quantities demanded for each good.
        """
        alphas = self.utility_function.coefficients
        total_alpha = np.sum(alphas)
        one_over_total_alpha = 1/total_alpha


        with np.errstate(divide='ignore', invalid='ignore'):
                # suppress divide by zero warning
                total_product = np.prod([
                    (prices[i] / alphas[i]) ** alphas[i]
                    for i in range(self.utility_function.num_inputs)
                ])
        
        hicksian_demand = [(alphas[i]/prices[i]) * (total_product * self.utility_level)**one_over_total_alpha for i in range(self.utility_function.num_inputs)]
        return hicksian_demand
    
    def _linear_utility(self, prices):
        # 3 GOODS MAX
        # DIFFICULTY LEVELS: 3 different levels (basis on prices given)
        """
        Computes the hicksian demand for goods given Linear utility function and prices.

        Parameters:
        - prices: A list of prices for each good.

        Returns:
        - demand: A list of (adjusted) quantities demanded for each good.
        """
        coeff = self.utility_function.coefficients
        utility_level  = self.utility_level
        coeff_to_price_ratio = np.array(coeff) / np.array(prices)
        max_ratio = np.max(coeff_to_price_ratio)
        consumption_indicators = (coeff_to_price_ratio == max_ratio)
        active_indices = np.where(consumption_indicators)[0]  # indices of goods to be consumed

        hicksian_demands = np.zeros(len(prices))

        remaining_utility = utility_level
        for i, idx in enumerate(active_indices):
            # if it's the last good being allocated ... 
            if i == len(active_indices) - 1:
                # amount needed so that a_1x_1 + a_2x_2 + ...  = utility_level
                hicksian_demands[idx] = remaining_utility / coeff[idx]
            else:
                max_consumption_for_good = remaining_utility / coeff[idx]

                allocation_for_good = legacy_np.uniform(0, max_consumption_for_good)
                hicksian_demands[idx] = allocation_for_good
                
                remaining_utility -= coeff[idx] * allocation_for_good
            
            if remaining_utility <= 0:
                break

        return hicksian_demands
    
    def get_quantity_at(self, prices):
        """
        Computes the quantity demanded for goods given the prices.

        Parameters:
        - prices: A list of prices for each good.

        Returns:
        - demand: A list of quantities demanded for each good.
        """
        if hasattr(self, 'utility_function'):
            if self.utility_function.family == 'cobb-douglas':
                return self._cobb_douglas_utility(prices)
            if self.utility_function.family == 'linear':
                return self._linear_utility(prices)
            else:
                return None

    def show_demand_function(self, utility_level):
        """
        Returns the functional form of the demand function.
        
        Returns:
        - demand_function: A sympy expression representing the demand function.
        """
        # Check that:
        # - utility function is a class attribute
        # - utility function only ha
        assert hasattr(self, 'utility_function'), "Must call from_utility_fn to derive the demand function."
        assert self.utility_function.family in ['cobb-douglas'], "Currently, only Cobb-Douglas."

        if self.utility_function.num_inputs == 2:
            prices = ["p_x", "p_y"]
        else:
            prices = [f"p_{i}" for i in range(1, self.utility_function.num_inputs + 1)] #["p_1", "p_2", ..."p_{n}"]
                
        if self.utility_function.family == 'cobb-douglas':
            
            alphas = self.utility_function.coefficients
            total_alpha = np.sum(alphas)
            one_over_total_alpha = 1/total_alpha
            
            demands = []
            total_product = 1

            # Suppress divide by zero warnings
            with np.errstate(divide='ignore', invalid='ignore'):
                for k in range(self.utility_function.num_inputs):
                    total_product = total_product*(((1/alphas[k])**alphas[k]))

            total_product_times_u = utility_level * total_product
            total__product_times_u_to_power = total_product_times_u**one_over_total_alpha

            for i in range(self.utility_function.num_inputs):
                
                prices_joined = " * ".join(
                    f"{prices[j]}^({alphas[j] / total_alpha - 1 if j == i else alphas[j] / total_alpha})"
                    for j in range(self.utility_function.num_inputs)
                )
                demand = f"{alphas[i] * total__product_times_u_to_power}({prices_joined})" 
                demands.append(demand)
            return demands
        else:
            return None
        

@function_registry.register('deriving_demand')
def gen_deriving_demand_utils(metadata):
    """
    Marshallian demand for a Cobb-Douglas utility U = x^a * y^b with income m:
        x = (a / (a + b)) m / p_x,   y = (b / (a + b)) m / p_y.
    good_1 asks for x, good_2 for y (the sampler rewrites good_2 templates to
    name good y). Every option has the form "<coefficient>/p_x" (review, October 2026: the old
    "coefficient * price" distractor repeated only the key's coefficient, which gave it away). The coefficients are
    the 2 x 2 grid {own exponent, other exponent} x {normalised by a + b, not normalised}:
        a m / (a + b) (key),  b m / (a + b),  a m,  b m      (for x; a and b swapped for y)
    so every coefficient is a natural mistake and each feature occurs in two options (no odd one out). Draws where
    two coefficients are within 3% of each other are redrawn. Computed from the shown values.
    """
    index = 0 if metadata['type'] == 'good_1' else 1
    var = 'x' if index == 0 else 'y'
    while True:
        a = round(float(rng.uniform(0.1, 0.9)), 2)
        b = round(float(rng.uniform(0.1, 0.9)), 2)
        budget = round(float(rng.uniform(100, 1000)), 2)
        exps = (a, b)
        coefs = [round(exps[index] / (a + b) * budget, 2), round(exps[1 - index] / (a + b) * budget, 2),
                 round(exps[index] * budget, 2), round(exps[1 - index] * budget, 2)]
        if all(_apart(c, coefs[:i]) for i, c in enumerate(coefs)) and not near_tie(exps[index] / (a + b) * budget):
            break
    answer = f"{coefs[0]}/p_{var}"
    options = [f"{c}/p_{var}" for c in coefs]
    answer_mask, options = gen_answer_and_options(answer, options=options, num_options=4)
    return {
        'budget': budget,
        'utility_func': f"U = x^{a} * y^{b}",
        'options_list': options,
        'correct_answer': answer_mask
    }

def _apart(v, others, rel=0.03, abs_=0.015):
    return all(abs(v - w) > max(abs_, rel * max(abs(v), abs(w))) for w in others)


def _scaled_distractors(answer, num_options=4):
    """The answer and num_options - 1 multiples of it (factor 0.5-2), each at least max(0.015, 3%) from the
    others after rounding to 2 decimals (independent multiples were sometimes 0.01 apart, e.g. 6.16 / 6.17)."""
    vals = [answer]
    for _ in range(500):
        if len(vals) == num_options:
            return vals
        c = answer * pyrng.uniform(0.5, 2)
        if _apart(round(c, 2), [round(v, 2) for v in vals]):
            vals.append(c)
    # tiny answers cannot be separated at 2 decimals; the sampler redraws on duplicate options
    return [answer] + [answer * f for f in (0.5, 1.5, 2.0)][:num_options - 1]


def linear_hicksian_options(coeffs, prices, u, k, want_bought, num_options=4):
    """
    Options for the Hicksian demand of good k under linear utility a.x with prices p and target utility u.

    Returns (option texts, one-hot mask) or (None, None) if the draw must be redone (near-tie between the best
    two goods, the asked good is / is not the bought one contrary to `want_bought`, or too few well-separated
    distractors). The bought good b maximises a_i/p_i; the key is u/a_k if k == b else 0. The other options are
    the quantities a solver gets from natural mistakes:
      - u/a_k            the asked good bought as if it were the cheapest per util (key when k == b)
      - u/a_b            the quantity of the good that is actually bought
      - u/(3 a_k)        an interior answer: each good supplies a third of the utility
      - u/a_j            the quantity if the second-best good j were bought
      - 0                always offered, so its presence gives nothing away
    All options are distinct after rounding to 2 decimals and kept at least max(0.015, 3%) apart; when the key is
    u/a_k, no distractor lies in the range u/a_k can take given the 2-dp rounding of a_k.
    """
    ratios = [a / p for a, p in zip(coeffs, prices)]
    order = sorted(range(3), key=lambda i: ratios[i], reverse=True)
    b, j = order[0], order[1]
    if ratios[b] - ratios[j] < 0.05 * ratios[b] or (k == b) != want_bought:
        return None, None
    key = u / coeffs[k] if k == b else 0.0
    cands = [u / coeffs[k], u / coeffs[b], u / (3 * coeffs[k]), u / coeffs[j]]
    vals = [round(key, 2)] + ([0.0] if key != 0 else [])
    # a reader who allows for the 2-dp rounding of the shown coefficients can only pin u/a_k down to this band
    band = (u / (coeffs[k] + 0.005) - 0.011, u / max(coeffs[k] - 0.005, 1e-9) + 0.011)
    for c in cands:
        c = round(c, 2)
        in_band = key != 0 and band[0] <= c <= band[1]
        if len(vals) < num_options and _apart(c, vals) and not in_band:
            vals.append(c)
    if len(vals) < num_options:
        return None, None
    texts = [str(v) for v in vals]
    order_idx = list(rng.permutation(len(texts)))
    texts = [texts[i] for i in order_idx]
    correct = [1 if i == 0 else 0 for i in order_idx]
    return texts, correct


def _r2(v):
    v = round(float(v), 2)
    return 0.0 if v == 0 else v


def hicksian_function_text(coef, exps, names):
    return f"{_r2(coef)}(" + " * ".join(f"{n}^({_r2(e)})" for n, e in zip(names, exps)) + ")"


def hicksian_cd_function_options(alphas, u, index):
    """Key and three distractors for the Hicksian demand of good `index` under U = prod x_k^alpha_k (functional
    form, prices as symbols). With S = sum alpha_k the key is
        C (prod_k p_k^(e_k)),  C = alpha_i (u prod_k (1/alpha_k)^alpha_k)^(1/S) * ... (as show_demand_function),
        e_i = alpha_i/S - 1,  e_k = alpha_k/S (k != i).
    Review, October 2026: the old distractors were the same function at other utility levels, so every
    option had the key's price exponents and setting all prices to 1 was enough. Now the options are the 2 x 2 grid
    {C, C'} x {e, e'}: e' is an exponent mistake (the exponents not divided by S; the price dependence inverted; the
    exponents of another good's demand) and C' a constant mistake (the alpha_i factor left out; the 1/S power left
    out of the utility term; the constant at a lower utility level), each chosen at random among those that differ
    from the key's after rounding to 2 decimals. Every option is a price-dependent function on the key's scale and
    each feature occurs in two options."""
    n = len(alphas)
    names = ["p_x", "p_y"] if n == 2 else [f"p_{k}" for k in range(1, n + 1)]
    S = float(sum(alphas))
    base = u * float(np.prod([(1 / a) ** a for a in alphas]))
    coef = alphas[index] * base ** (1 / S)
    exps = [alphas[k] / S - (1 if k == index else 0) for k in range(n)]
    exp_mistakes = [
        [alphas[k] - (1 if k == index else 0) for k in range(n)],
        [-e for e in exps],
    ] + [[alphas[k] / S - (1 if k == j else 0) for k in range(n)] for j in range(n) if j != index]
    coef_mistakes = [base ** (1 / S), alphas[index] * base, coef * float(rng.uniform(0.3, 0.8)) ** (1 / S)]

    # an exponent mistake must change some shown exponent by at least 0.05: with a degree S near 1, leaving out
    # the division by S moved an exponent by 0.01 (-0.13 vs -0.14), a second correct option for any reader
    exp_mistakes = [e for e in exp_mistakes if max(abs(_r2(a) - _r2(b)) for a, b in zip(e, exps)) >= 0.05]
    coef_mistakes = [c for c in coef_mistakes if 0.05 <= c <= 1e6 and abs(c - coef) >= 0.05 * max(c, coef)]
    if not exp_mistakes or not coef_mistakes:
        return None
    e2 = exp_mistakes[int(rng.integers(0, len(exp_mistakes)))]
    c2 = coef_mistakes[int(rng.integers(0, len(coef_mistakes)))]
    opts = [hicksian_function_text(c, e, names) for c in (coef, c2) for e in (exps, e2)]
    if len(set(opts)) != 4:
        return None
    return opts[0], opts


def _hicksian_cd_utility(n):
    """Cobb-Douglas utility with every exponent at least 0.1 and degree at least 0.5 (exponents such as 0.07 and
    0.08 gave Hicksian demands of 1e19, printed in scientific notation)."""
    while True:
        u = UtilityFunction(num_inputs=n, family='cobb-douglas')
        if min(u.coefficients) >= 0.1 and sum(u.coefficients) >= 0.5:
            return u


@function_registry.register('deriving_hicksian_demand')
def gen_deriving_hicksian_demand_values(metadata):
    # This should always be a cobb-douglas utility function
    # difficulties are:
    #   - 0: # 2 goods, both prices are given (numerical form)
    #   - 1: # 2 goods, no prices are given (functional form)
    #   - 2: # 3 goods, no prices are given (functional form)
    # types are: good_1 and good_2 (and good_3 for difficulty 2 only)
    # Can either directly create demand functions or from a random utility function.
    family = 'cobb-douglas'
    if metadata['difficulty_level'] == 0: # 2 goods, both prices are given (numerical form), HICKSIAN
        if metadata['type'] == 'good_1':
            index = 0
        if metadata['type'] == 'good_2':
            index = 1
        
        utility_function = _hicksian_cd_utility(2)
        utility_level = np.round(pyrng.uniform(100, 1000),2)
        hicksian_demand_function = HicksianDemandFunction.from_utility_fn(utility_function, utility_level)
        prices = [round(pyrng.uniform(1, 10), 2) for _ in range(2)]

        answers = hicksian_demand_function.get_quantity_at(prices)
        answer = answers[index]
        if not 0.05 <= answer <= 1e6:   # printed as 3.83e+19 or 0.0
            return gen_deriving_hicksian_demand_values(metadata)


        options = _scaled_distractors(answer)
        answer_mask,options = gen_answer_and_options(answer,options=options,num_options=4)

        return {
            'utility_level': utility_level,
            'utility_func': str(utility_function),
            'price_1': prices[0],
            'price_2': prices[1],
            'options_list': options,
            'correct_answer': answer_mask
        }

    if metadata['difficulty_level'] == 1: # 2 goods, no prices are given (functional form), HICKSIAN
        
        if metadata['type'] == 'good_1': #return hicksian demand of 1st good
            index = 0
        if metadata['type'] == 'good_2': #return hickisan demand of 2nd good
            index = 1

        while True:
            utility_function = _hicksian_cd_utility(2)
            utility_level = np.round(pyrng.uniform(100, 1000),2)
            got = hicksian_cd_function_options(list(utility_function.coefficients), utility_level, index)
            if got is not None:
                break
        answer, options = got
        answer_mask,options = gen_answer_and_options(answer,options=options,num_options=4)
        return {
            'utility_level': utility_level,
            'utility_func': str(utility_function),
            'options_list': options,
            'correct_answer': answer_mask
        }

    if metadata['difficulty_level'] == 2: #3 goods, no prices are given (functional form), HICKSIAN
        if metadata['type'] == 'good_1': #return hicksian demand of 1st good
            index = 0
        if metadata['type'] == 'good_2': #return hickisan demand of 2nd good
            index = 1
        if metadata['type'] == 'good_3': #return hickisan demand of 3rd good
            index = 2

        while True:
            utility_function = _hicksian_cd_utility(3)
            utility_level = np.round(pyrng.uniform(100, 1000),2)
            got = hicksian_cd_function_options(list(utility_function.coefficients), utility_level, index)
            if got is not None:
                break
        answer, options = got
        answer_mask,options = gen_answer_and_options(answer,options=options,num_options=4)
        return {
            'utility_level': utility_level,
            'utility_func': str(utility_function),
            'options_list': options,
            'correct_answer': answer_mask
        }

    if metadata['difficulty_level'] == 3: # 3 goods, prices given, LINEAR
        family = 'linear'
        if metadata['type'] == 'good_1': #return hicksian demand of 1st good
            index = 0
        if metadata['type'] == 'good_2': #return hickisan demand of 2nd good
            index = 1
        if metadata['type'] == 'good_3': #return hickisan demand of 3rd good
            index = 2
        
        # Linear utility: the consumer buys only the good with the highest coefficient/price ratio, so the
        # Hicksian demand for the asked good is u/a_k if it is that good and 0 otherwise. Coefficients are
        # rounded to the shown 2 decimals (and >= 0.05) before solving; near-ties between goods are redrawn.
        # Whether the asked good is the one bought is a fair coin, so the key is 0 in about half the questions.
        want_bought = bool(rng.random() < 0.5)
        while True:
            utility_function = UtilityFunction(num_inputs=3,family=family)
            utility_function.coefficients = np.round(rng.uniform(0.05, 1, 3), 2)
            utility_level = round(float(rng.uniform(100, 1000)), 2)
            prices = [round(float(rng.uniform(1, 10)), 2) for _ in range(3)]
            options, correct = linear_hicksian_options(list(utility_function.coefficients[:3]), prices,
                                                       utility_level, index, want_bought)
            if options is not None:
                break
        answer_mask, options = [correct], [options]

        utility_function_str = str(utility_function)
        utility_function_str = utility_function_str[:utility_function_str.rfind("+")] if "+" in utility_function_str else utility_function_str
        utility_function_str = utility_function_str.rstrip()   # "0.91 * x3 , where" (the cut left a space)

        return {
            'utility_level': utility_level,
            'utility_func': utility_function_str,
            'price_1': prices[0],
            'price_2': prices[1],
            'price_3': prices[2],
            'options_list': options,
            'correct_answer': answer_mask
        }
    
    if metadata['difficulty_level'] == 4: # 3 goods, 3 prices are given (numerical form), HICKSIAN
        if metadata['type'] == 'good_1':
            index = 0
        if metadata['type'] == 'good_2':
            index = 1
        if metadata['type'] == 'good_3':
            index = 2
        
        utility_function = UtilityFunction(num_inputs=3,family=family)
        utility_level = np.round(pyrng.uniform(100, 1000),2)
        hicksian_demand_function = HicksianDemandFunction.from_utility_fn(utility_function, utility_level)
        prices = [round(pyrng.uniform(1, 10), 2) for _ in range(3)]

        answers = hicksian_demand_function.get_quantity_at(prices)
        answer = answers[index]


        options = _scaled_distractors(answer)
        answer_mask,options = gen_answer_and_options(answer,options=options,num_options=4)

        return {
            'utility_level': utility_level,
            'utility_func': str(utility_function),
            'price_1': prices[0],
            'price_2': prices[1],
            'price_3': prices[2],
            'options_list': options,
            'correct_answer': answer_mask
        }
    
# engel_curves: replaced by filler_tools/engel_curves_tools.py (code templates), 5 October 2026.


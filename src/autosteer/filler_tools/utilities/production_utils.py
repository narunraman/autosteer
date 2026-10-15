import numpy as np
from scipy.optimize import minimize
from .filler_utils import rng
from .mistake_options import pick_mistakes, options_from_values
from autosteer.filler_tools.rng import legacy_np, pyrng
'''
Samples a production function (a function of two variables L and K) 
from a given family

familys should probably be at least linear, quadratic, cobb-douglas
when generate ensure that the function is increasing in K and L for at least some beginning region
ideally production fn satifies diminishing marginal returns 
'''
class ProductionFunction:
    # Note: K,L here represent coefficients
    def __init__(self, family: str,max_coeff=None):
        default_max_coeffs = {'Cobb_Douglas':2,'linear':10,'dynamic':5,'Leonteif':5}
        self.family = family
        self.prod_func = {}
        if max_coeff is None:
            max_coeff = default_max_coeffs[family]
        if family == 'linear':
            self.L = round(legacy_np.rand()*10,2)
            self.K = round(legacy_np.rand()*10,2)
            self.function_string = f"{self.L}L + {self.K}K"
        elif family == 'Cobb_Douglas':
            self.L = round(legacy_np.rand()*max_coeff,2)
            self.K = round(legacy_np.rand()*max_coeff+0.01,2)
            self.function_string = f"L^{self.L}K^{self.K}"
        elif family == 'dynamic':
            self.K = round(legacy_np.rand()*max_coeff,2)
            self.alpha = round(legacy_np.rand()*max_coeff*5,2)
            self.function_string = f"{self.alpha}K^{self.K}"
        elif family == 'Leonteif':
            self.L = round(legacy_np.rand()*max_coeff+1,2)
            self.K = round(legacy_np.rand()*max_coeff+1,2)
            self.v = 0.5
            self.function_string = f"min(L/{self.L},K/{self.K})^{self.v}"
        
    def profit(self,labor,capital,price,lcost,kcost=None):
        if self.family == 'Cobb_Douglas':
            if kcost is None:
                profit = price*(labor**self.L*capital**self.K) - labor*lcost # this was here before, wanted to make sure I wasn't breaking any of the code
            else:
                profit = price*(labor**self.L*capital**self.K) - labor*lcost - capital*kcost  
        elif self.family == 'Leonteif':
            capital = (self.K*labor)/self.L
            profit = price*min(labor/self.L,capital/self.K)**self.v-labor*lcost-capital*kcost
        return profit
    
    def find_profit_max(self,price,capital,lcost,kcost=None):
        # find the price where quantity demanded equals quantity supplied
        profit = lambda x: -self.profit(x,capital,price,lcost,kcost)
        result = minimize(profit, x0 = np.array([legacy_np.rand()]))
        if result.success:
            return result.x
        else:
            return None
        
    def production_output(self, labor, capital):
        # HELPER FOR find_cost_min(.)
        # TODO: document method
        if self.family == 'Cobb_Douglas':
            return (labor ** self.L) * (capital ** self.K)
        elif self.family == 'Leonteif':
            return min(labor / self.L, capital / self.K)**self.v
        elif self.family == 'linear': 
            return (labor * self.L) + (capital * self.K)
        else:
            raise ValueError("Unsupported production function type")
    
    def linear_cost(self, labor, capital, lcost, kcost):
        # HELPER FOR find_cost_min(.)
        # TODO: document method
        return labor * lcost + capital * kcost
    
    def find_cost_min(self, Y, lcost, kcost, cost_function = "linear"):
        # TODO: document method
        if cost_function == "linear":
            cost = lambda inputs: inputs[0] * lcost + inputs[1] * kcost  # inputs[0] = L, inputs[1] = K
        else:
            cost = lambda inputs: (inputs[0]**2)*lcost + (inputs[1]**2)*kcost

       # f(L, K) - Y >= 0
        constraint = {
            'type': 'ineq', 
            'fun': lambda inputs: self.production_output(inputs[0], inputs[1]) - Y
        }

        result = minimize(
            cost,
            x0=np.array([legacy_np.rand(), legacy_np.rand()]),  # Initial guesses for L (Labour) and K (Capital)
            constraints=[constraint],
            bounds=[(0, None), (0, None)]  # L, K >= 0 
        )

        if result.success:
            return result.x  # Optimal labor (L) and capital (K)
        else:
            return None  # No valid solution found
    
    def find_labor_demand(self, Y, kcost, lcost, capital_prime):
        """
        TODO: documentation
        """
        if self.family != "Cobb_Douglas":
            raise NotImplementedError("only implemented for Cobb-Douglas.")

        labor = (Y / (capital_prime ** self.K)) ** (1 / self.L)
        return labor if labor >= 0 else None  # labor demand is non-negative

    def find_capital_demand(self, Y, kcost, lcost, labour_prime):
        """
        TODO: documentation
        """
        if self.family != "Cobb_Douglas":
            raise NotImplementedError("only implemented for Cobb-Douglas.")

        capital = (Y / (labour_prime ** self.L)) ** (1 / self.K)
        return capital if capital >= 0 else None  # labor demand is non-negative
    
    def set_and_find_TFP(self, labor, capital):
        """
        :EFFECT: set and compute Total Factor Productivity (TFP)
        
        TODO: properly document method
        """
        self.A = round(legacy_np.rand()*10+0.01,2)
        total_output = self.A * self.production_output(labor, capital)
        return [self.A, total_output]

    
    
def sample_prod_function(family='linear',max_coeff=2, concave=False):
    """
    Sample a production function of two inputs L and K.

    concave: for 'quadratic' only, draw negative L^2 and K^2 coefficients so
    the marginal products are strictly decreasing (default: non-negative).
    """
    prod_func = {}
    if family.lower() == 'linear':
        # coefficients at least 0.5 (a draw rounding to 0 printed "0.0K")
        prod_func['L'] = round(0.5 + legacy_np.rand()*9.5,2)
        prod_func['K'] = round(0.5 + legacy_np.rand()*9.5,2)
        prod_func['function_string'] = f"{prod_func['L']}L + {prod_func['K']}K"
        prod_func['function'] = lambda L,K: prod_func['L']*L + prod_func['K']*K
    elif family.lower() == 'cobb_douglas':
        # exponents at least 0.1 (draws rounding to 0 or 0.01 printed "L^0.0" and gave zero coefficients)
        prod_func['L'] = round(0.1 + legacy_np.rand()*(max_coeff - 0.1),2)
        prod_func['K'] = round(0.1 + legacy_np.rand()*(max_coeff - 0.1),2)
        prod_func['function_string'] = f"L^{prod_func['L']}K^{prod_func['K']}"
        prod_func['function'] = lambda L,K: L**prod_func['L']*K**prod_func['K']
    elif family.lower() == 'dynamic':
        prod_func['K'] = round(0.1 + legacy_np.rand()*(max_coeff - 0.1),2)
        prod_func['alpha'] = round(0.5 + legacy_np.rand()*(max_coeff*5 - 0.5),2)
        prod_func['function_string'] = f"{prod_func['alpha']}K^{prod_func['K']}"
    elif family.lower() == 'quadratic':
        prod_func['a'] = round(legacy_np.rand()*max_coeff,2)
        prod_func['b'] = round(legacy_np.rand()*max_coeff,2)
        if concave:
            # strictly negative squares; scaled down so MP stays positive over a useful range
            prod_func['c'] = -round(legacy_np.rand()*max_coeff/10+0.01,2)
            prod_func['d'] = -round(legacy_np.rand()*max_coeff/10+0.01,2)
        else:
            prod_func['c'] = round(legacy_np.rand()*max_coeff,2)
            prod_func['d'] = round(legacy_np.rand()*max_coeff,2)
        prod_func['e'] = round(legacy_np.rand()*max_coeff,2)
        prod_func['L'] = [prod_func['a'],prod_func['c'],prod_func['e']]
        prod_func['K'] = [prod_func['b'],prod_func['d'], prod_func['e']]
        var_list = ['L', 'K', 'L^2', 'K^2', 'LK']
        # Show every non-zero term (previously only the last term was kept).
        terms = []
        for i, key in enumerate(['a','b','c','d','e']):
            coef = prod_func[key]
            if coef == 0:
                continue
            if not terms:
                terms.append(f"{coef}{var_list[i]}")
            elif coef < 0:
                terms.append(f"- {abs(coef)}{var_list[i]}")
            else:
                terms.append(f"+ {coef}{var_list[i]}")
        prod_func['function_string'] = ' '.join(terms) if terms else '0'
        prod_func['function'] = lambda L,K: prod_func['a']*L + prod_func['b']*K + prod_func['c']*L**2 + prod_func['d']*K**2 + prod_func['e']*L*K
    return prod_func

def sample_cost_function(family='linear', max_coeff=2):
    cost_func = {}
    if family == 'linear':
        cost_func['a'] = round(legacy_np.rand()*10,2)
        cost_func['b'] = round(legacy_np.rand()*10,2)
        cost_func['function_string'] = f"{cost_func['a']}Q + {cost_func['b']}"
        cost_func['function'] = lambda x: cost_func['a']*x + cost_func['b']
        cost_func['marginal_function'] = lambda x: cost_func['a']
    elif family == 'polynomial':
        # the largest power is set by max_coeff
        cost_func['coeffs'] = [round(legacy_np.rand()*max_coeff,2) for i in range(max_coeff)]
        cost_func['function_string'] = ''
        for i in range(1, max_coeff):
            if cost_func['coeffs'][i] == 0:
                continue
            cost_func['function_string'] += f"{cost_func['coeffs'][i]}Q^{i+1} + "
        constant = round(legacy_np.rand()*10,2)
        cost_func['function_string'] += f"{cost_func['coeffs'][-1]}Q + {constant}"

        # cost_func['function_string'] = ' + '.join([f"{cost_func['coeffs'][i]}Q^{i+1}" if cost_func['coeffs'][i] > 0 else '' for i in range(max_coeff)]) + f" + {round(np.random.rand()*10,2)}"
        cost_func['function'] = lambda x: sum([cost_func['coeffs'][i]*x**i for i in range(max_coeff-1, 0, -1)]) + constant
        cost_func['marginal_function'] = lambda x: sum([cost_func['coeffs'][i]*i*x**(i-1) for i in range(1,max_coeff)])

    return cost_func

def sample_supply_function(family='linear',max_coeff=2):
    supply_func = {}
    if family == 'linear':
        # slope and intercept at least 0.5: a zero intercept ("Q = 7.2P + 0.0") makes the elasticity 1 trivially
        supply_func['a'] = round(0.5 + legacy_np.rand()*9.5,2)
        supply_func['b'] = round(0.5 + legacy_np.rand()*9.5,2)
        # quantity supplied as a function of price (the elasticity formulas
        # in supply_elasticity differentiate with respect to P)
        supply_func['function_string'] = f"Q = {supply_func['a']}P + {supply_func['b']}"
    elif family == 'quadratic':
        supply_func['a'] = round(0.05 + legacy_np.rand()*(max_coeff - 0.05),2)
        supply_func['b'] = round(0.05 + legacy_np.rand()*(max_coeff - 0.05),2)
        supply_func['c'] = round(0.05 + legacy_np.rand()*(max_coeff - 0.05),2)
        supply_func['function_string'] = f"Q = {supply_func['a']}P^2 + {supply_func['b']}P + {supply_func['c']}"
    return supply_func

def sample_agg_demand_function(family='linear',max_coeff=2):
    supply_func = {}
    if family == 'linear':
        supply_func['a'] = round(legacy_np.rand()*10,2)
        supply_func['b'] = round(legacy_np.rand()*10,2)
        supply_func['function_string'] = f"{supply_func['a']}P + {supply_func['b']}"
    elif family == 'quadratic':
        supply_func['a'] = round(legacy_np.rand()*max_coeff,2)
        supply_func['b'] = round(legacy_np.rand()*max_coeff,2)
        supply_func['c'] = round(legacy_np.rand()*max_coeff,2)
        supply_func['function_string'] = f"{supply_func['a']}P^2 + {supply_func['b']}P + {supply_func['c']}"
    return supply_func

def find_profit_max_CD(prod_func,cost,price,capital):
    # find the price where quantity demanded equals quantity supplied
    def profit(x):
        return -CD_profit(x, prod_func, cost, price, capital)
    result = minimize(profit, x0 = np.array([legacy_np.rand()]))
    if result.success:
        return result.x
    else:
        return None
    
def CD_profit(labor,prod_func,cost,price,capital):
    return price*(labor**prod_func['L']*capital**prod_func['K']) - labor*cost

def find_profit_max_dyn(capital_1,discount,prod_func,price_1,price_2):
    # find the price where quantity demanded equals quantity supplied
    def profit(x):
        return -dynamic_profit(x, capital_1, discount, prod_func, price_1, price_2)
    result = minimize(profit, x0 = np.array([legacy_np.rand()]))
    if result.success:
        return result.x
    else:
        return None

def _opt_K2(cap1, disc, prod_fn, price1, price2):
    """Return profit-maximising K2 for a given tomorrow price."""
    def neg_profit(K2):
        prod_today     = (cap1 ** prod_fn['K']) * prod_fn['alpha']
        prod_tomorrow  = (K2   ** prod_fn['K']) * prod_fn['alpha']
        π = ( price1 * prod_today
              - (K2 - cap1) ** 2
              + disc * price2 * prod_tomorrow )
        return -π
    res = minimize(neg_profit, x0=np.array([cap1]), bounds=[(0, None)],
                   method='L-BFGS-B')
    # res.x is a 1-element array; float() of it is an error from numpy 2.4 on
    return np.asarray(res.x).item(), -np.asarray(res.fun).item()          # (arg-max, max value)
    
def dynamic_profit(capital,capital_1,discount,prod_func,price_1,price_2):
        return price_1*((capital_1**prod_func['K'])*prod_func['alpha']) - (capital-capital_1)**2+discount*(price_2*((capital**prod_func['K'])*prod_func['alpha']))


def _pct_change_options(answer_pct):
    """
    Options for a percentage price change: the answer, the same size in the
    wrong direction, no change, and an overshoot. Rendered like '-10.71%'.
    """
    answer_pct = round(float(answer_pct), 2)
    overshoot = round(answer_pct * float(rng.uniform(1.5, 2.5)), 2)
    values = [answer_pct, -answer_pct, 0.0, overshoot]
    options = [f"{v:.2f}%" for v in values]
    answer = options[0]
    pyrng.shuffle(options)
    answer_mask = [1 if o == answer else 0 for o in options]
    return [answer_mask], [options]


# TFP / input-price shocks. With constant returns Q = A L^a K^(1-a) and
# competitive markets, the long-run equilibrium price equals unit cost
#     P = c(w, r) / A = (1/A) (w/a)^a (r/(1-a))^(1-a).
# Hence a productivity change of x% moves P by 1/(1 + x) - 1, and a wage
# change of x% moves P by (1 + x)^a - 1. (The previous answers, -x and a*x,
# were the first-order approximations.)

def _tfp_options(key_pct, share_mistakes, other_mistakes):
    """
    Review (October 2026): the options were the key, its negative, 0 and an overshoot, so the one value whose
    negative was also offered was the key (given the direction). Now one mistake with an input share applied the
    wrong way (these cluster together, and three of them would leave the key as the odd one out) and two of the
    others: the first-order approximation or full pass-through, no change (0), the effect in the wrong direction.
    Never two values of opposite sign and equal size.
    """
    kw = dict(same_sign=False, no_negation=True, allow_zero=True, rel=0.03, absolute=0.15, max_factor=20)
    for _ in range(50):
        first = pick_mistakes(key_pct, share_mistakes, n=1, fill=False, **kw)
        if first is None:
            break
        rest = pick_mistakes(key_pct, [m for m in other_mistakes if abs(m + first[0]) > 0.15 and
                                       abs(m - first[0]) > max(0.15, 0.03 * abs(m))], n=2, fill=False, **kw)
        if rest is not None:
            return options_from_values(key_pct, first + rest, fmt=lambda v: f"{round(float(v), 2):.2f}%")
    d = pick_mistakes(key_pct, share_mistakes + other_mistakes, **kw)
    return options_from_values(key_pct, d, fmt=lambda v: f"{round(float(v), 2):.2f}%")


def _tfp_productivity_options(g, alpha):
    """g: signed productivity change (0.09 for +9%). Key 1/(1+g) - 1."""
    key = (1 / (1 + g) - 1) * 100
    share = [-alpha * g, -(1 - alpha) * g, (1 + g) ** -alpha - 1, (1 + g) ** -(1 - alpha) - 1]
    other = [-g, 0.0, g, 1 / (1 - g) - 1]
    return _tfp_options(key, [x * 100 for x in share], [x * 100 for x in other])


def _tfp_wage_options(g, alpha):
    """g: signed wage change. Key (1+g)^alpha - 1 (alpha the labor exponent)."""
    key = ((1 + g) ** alpha - 1) * 100
    share = [alpha * g, (1 - alpha) * g, (1 + g) ** (1 - alpha) - 1]
    other = [g, 0.0, -g, (1 + g) ** -alpha - 1, -alpha * g]
    return _tfp_options(key, [x * 100 for x in share], [x * 100 for x in other])


def gen_prod_increase():
    alpha = round(float(rng.uniform(0.4, 0.6)), 2)
    beta = round(1 - alpha, 2)
    productivity_increase = int(rng.integers(5, 15))  # 5% to 14%
    answer_mask, options_list = _tfp_productivity_options(productivity_increase / 100, alpha)
    return {
        'prod_fn': f"Q = A * L^{alpha} * K^{beta}",
        'prod_level': 'A',
        'productivity_increase': f"{productivity_increase}%",
        'options_list': options_list,
        'correct_answer': answer_mask,
    }

def gen_prod_decrease():
    alpha = round(float(rng.uniform(0.4, 0.6)), 2)
    beta = round(1 - alpha, 2)
    productivity_decrease = int(rng.integers(5, 15))  # 5% to 14%
    answer_mask, options_list = _tfp_productivity_options(-productivity_decrease / 100, alpha)
    return {
        'prod_fn': f"Q = A * L^{alpha} * K^{beta}",
        'prod_level': 'A',
        'productivity_decrease': f"{productivity_decrease}%",
        'options_list': options_list,
        'correct_answer': answer_mask,
    }

def gen_labor_increase():
    alpha = round(float(legacy_np.uniform(0.4, 0.6)), 2)
    labor_price_increase = int(legacy_np.randint(5, 15))  # 5% to 14%
    answer_mask, options_list = _tfp_wage_options(labor_price_increase / 100, alpha)
    return {
        'prod_fn': f"Q = A * L^{alpha} * K^{round(1 - alpha, 2)}",
        'prod_level': 'A',
        'labor_price_increase': f"{labor_price_increase}%",
        'options_list': options_list,
        'correct_answer': answer_mask,
    }

def gen_labor_decrease():
    alpha = round(float(legacy_np.uniform(0.4, 0.6)), 2)
    labor_price_decrease = int(legacy_np.randint(5, 15))  # 5% to 14%
    answer_mask, options_list = _tfp_wage_options(-labor_price_decrease / 100, alpha)
    return {
        'prod_fn': f"Q = A * L^{alpha} * K^{round(1 - alpha, 2)}",
        'prod_level': 'A',
        'labor_price_decrease': f"{labor_price_decrease}%",
        'options_list': options_list,
        'correct_answer': answer_mask,
    }

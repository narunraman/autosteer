import numpy as np
from random import shuffle
import sympy as sp
from scipy.optimize import minimize, Bounds, root_scalar
from .utilities.production_utils import ProductionFunction,sample_prod_function,CD_profit,find_profit_max_CD,gen_answer_and_options,sample_supply_function
from .fill_registry import FillRegister
from .utilities.filler_utils import rng
from .utilities.filler_utils import rng, gen_answer_and_options
import re
from .utilities.mistake_options import mistake_options, pick_mistakes, options_from_values

function_registry = FillRegister()

@function_registry.register('marginal_cost')
def gen_marginal_cost(metadata):
    family = metadata['type']
    if not family:
        family = 'Cobb_Douglas'
    # Redrawn until every shown number is at least 0.05 (capital, wage and coefficients could round to 0.0) and the
    # options are distinct; the wage never ends in a 0 cent (the sampler would print "8.4").
    while True:
        prod_fn = sample_prod_function(family)
        cost = round(np.random.rand()*9.5+0.5,2)
        capital = round(np.random.rand()*9.5+0.5,2)
        fixed_cost = round(np.random.rand()*100+1,2)
        if family == 'linear':
            marginal_cost = round(cost/prod_fn['L'],2)
            options = [prod_fn['L'],prod_fn['K'],round(marginal_cost*np.random.rand(),2),marginal_cost]
        else:
            # VC(Q) = w K^(-b/a) Q^(1/a); MC = (w/a) K^(-b/a) Q^(1/a - 1).
            # 1/a is used exactly (it was rounded to 2 dp inside the coefficient).
            inv_a = 1/prod_fn['L']
            expon = round(inv_a,2)
            coefficient = round(((1/(capital**prod_fn['K']))**inv_a)*cost/prod_fn['L'],2)
            marginal_cost = f"{coefficient}Q^{round(inv_a-1,2)}"
            options = [f"{prod_fn['L']}Q^{expon}",f"{prod_fn['K']}Q^{round(expon-1,2)}",f"{round(coefficient*np.random.choice([np.random.uniform(0.3, 0.8), np.random.uniform(1.25, 2)]),2)}Q^{round(expon-1,2)}",marginal_cost]
        # (and not 'Q^0.0': an exponent 1/a - 1 rounding to 0)
        coef_ok = family == 'linear' or (0.05 <= coefficient <= 1000 and abs(inv_a - 1) >= 0.05)
        if coef_ok and len({str(o) for o in options}) == len(options) and round(cost * 100) % 10:
            break
    answer = marginal_cost
    answer_mask,options = gen_answer_and_options(answer,options=options)
    return {
        "f_cost": fixed_cost,
        "p_function": prod_fn['function_string'],
        "capital": capital,
        "c_function": f"{cost}L",
        "wage": f"{cost:.2f}",
        "options_list":options,
        "correct_answer": answer_mask
    }

@function_registry.register('average_cost')
def gen_average_cost(metadata):
    family = metadata['type']
    if not family:
        family = 'Cobb_Douglas'
    # Redrawn until every shown number is at least 0.05 (capital, wage and coefficients could round to 0.0) and the
    # options are distinct; the wage never ends in a 0 cent (the sampler would print "8.4").
    while True:
        prod_fn = sample_prod_function(family)
        cost = round(np.random.rand()*9.5+0.5,2)
        capital = round(np.random.rand()*9.5+0.5,2)
        fixed_cost = round(np.random.rand()*100+1,2)
        if family == 'linear':
            average_cost = round(cost/prod_fn['L'],2)
            options = [prod_fn['L'],prod_fn['K'],round(average_cost*np.random.rand(),2),average_cost]
        else:
            # AC(Q) = w K^(-b/a) Q^(1/a - 1) + F/Q, with 1/a used exactly
            inv_a = 1/prod_fn['L']
            expon = round(inv_a,2)
            coefficient = round(((1/(capital**prod_fn['K']))**inv_a)*cost,2)
            average_cost = f"{coefficient}Q^{round(inv_a-1,2)}+{fixed_cost}/Q"
            options = [f"{prod_fn['L']}Q^{expon}+{fixed_cost}/Q",f"{prod_fn['K']}Q^{expon}",f"{round(coefficient*np.random.choice([np.random.uniform(0.3, 0.8), np.random.uniform(1.25, 2)]),2)}Q^{round(expon-1,2)}+{fixed_cost}/Q",average_cost]
        # (and not 'Q^0.0': an exponent 1/a - 1 rounding to 0)
        coef_ok = family == 'linear' or (0.05 <= coefficient <= 1000 and abs(inv_a - 1) >= 0.05)
        if coef_ok and len({str(o) for o in options}) == len(options) and round(cost * 100) % 10:
            break
    answer = average_cost
    answer_mask,options = gen_answer_and_options(answer,options=options)
    return {
        "f_cost": fixed_cost,
        "p_function": prod_fn['function_string'],
        "capital": capital,
        "c_function": f"{cost}L",
        "wage": f"{cost:.2f}",
        "options_list":options,
        "correct_answer": answer_mask
    }

@function_registry.register('returns_to_scale')
def gen_returns_to_scale(metadata):
    family = metadata['type']
    if not family:
        family = 'Cobb_Douglas'
    prod_fn = sample_prod_function(family)
    cost = round(np.random.rand()*10,2)
    capital = round(np.random.rand()*10,2)
    fixed_cost = round(np.random.rand()*100,2)
    rts=0
    if family == 'linear':
        rts = "Returns to scale are Constant"
        options = [rts,"Returns to scale are Increasing","Returns to scale are Decreasing"]
    elif family == 'Cobb_Douglas':
        if prod_fn['L']+prod_fn['K']>1:
            rts = "Returns to scale are Increasing"
            options = [rts,"Returns to scale are Decreasing","Returns to scale are Constant"]
        elif prod_fn['L']+prod_fn['K']<1:
            rts = "Returns to scale are Decreasing"
            options = [rts,"Returns to scale are Increasing","Returns to scale are Constant"]
        else:   
            rts = "Returns to scale are Constant"
            options = [rts,"Returns to scale are Increasing","Returns to scale are Decreasing"]
    # NOTE: I added this because there was an error.
    else:
        rts = "Unknown"
        options = [rts,"Increasing","Decreasing"]
    answer = rts
    answer_mask,options = gen_answer_and_options(answer,options=options,num_options=3)
    return {
        "f_cost": fixed_cost,
        "p_func": prod_fn['function_string'],
        "capital": capital,
        "c_function": f"{cost}L",
        "options_list":options,
        "correct_answer": answer_mask
    }

@function_registry.register('output_elasticity')
def gen_output_elasticity(metadata):
    family = metadata['type']
    prod_fn = sample_prod_function(family)
    capital = round(np.random.rand()*9.5+0.5,2)   # K = 0.0 made the linear key equal to the '1' distractor
    if family == 'linear':
        capital = int(np.random.randint(2, 11))   # bK then has 2 decimals and is shown exactly (Review)
    elasticity = 0
    if family == 'linear':
        # Review: bK is exact (K is a whole number; bK was rounded to 2 decimals unannounced), and the distractors are
        # dimensionless ratios from mistakes (the share of K, K left out of bK, the ratio inverted, the marginal
        # product a over output), so the key is not the only ratio and the only one with K substituted.
        a_, bk = prod_fn['L'], round(prod_fn['K'] * capital, 2)
        elasticity = f"{a_}L/({a_}L+{bk})"
        mistakes = [f"{bk}/({a_}L+{bk})", f"{a_}L/({a_}L+{prod_fn['K']})", f"({a_}L+{bk})/({a_}L)",
                    f"{a_}/({a_}L+{bk})"]
        options = [elasticity] + [mistakes[int(i)] for i in rng.permutation(len(mistakes))[:3]]
    elif family == 'Cobb_Douglas':
        # elasticity of output w.r.t. L is the exponent a. Distractors (1, a
        # random fraction of a, the K exponent) must differ numerically from a:
        # with a = 1.0 the '1' distractor was a second correct option.
        elasticity = round(prod_fn['L'],2)
        options = [elasticity]
        for cand in (1.0, prod_fn['K'], round(prod_fn['L'] + prod_fn['K'], 2), 0.5,
                     round(np.random.rand()*prod_fn['L'], 2), 2.0):
            if len(options) < 4 and all(abs(float(cand) - float(o)) > 0.005 for o in options):
                options.append(cand)
    answer = elasticity
    answer_mask,options = gen_answer_and_options(answer,options=options,num_options=4)
    return {
        "fix_var": "capital (K)",
        "p_function": prod_fn['function_string'],
        "fixed_val": f"K={capital}",
        "query_var": f"Labor (L)",
        "options_list":options,
        "correct_answer": answer_mask
    }

@function_registry.register('profit_maximization')
def gen_profit_maximization(metadata):
    """
    Short-run profit maximisation with capital fixed at K:
        max_L  p F(L, K) - w L
    Cobb-Douglas F = L^a K^b with 0 < a < 1 (draws with a >= 1 have no
    finite optimum and are rejected): L* = (a p K^b / w)^(1 / (1 - a)).
    Leontief F = min(L/cl, K/ck)^v, v = 0.5: L* = min(interior, cl K / ck)
    with interior = (v p / (w cl^v))^(1 / (1 - v)).
    The price of capital is not shown in the templates, so profit is
    revenue minus labor cost. Computed from the displayed (rounded) values.
    """
    while True:
        family = str(rng.choice(['Cobb_Douglas', 'Leonteif']))
        prod_fn = ProductionFunction(family)
        capital = round(float(rng.uniform(0.5, 10)), 2)
        lcost = round(float(rng.uniform(0.5, 10)), 2)
        price = round(float(rng.random()*10) + lcost, 2)
        if family == 'Cobb_Douglas':
            a, b = prod_fn.L, prod_fn.K
            if not 0.05 <= a <= 0.9:
                continue
            opt_labor = (a * price * capital ** b / lcost) ** (1 / (1 - a))
            output = opt_labor ** a * capital ** b
        else:
            cl, ck, v = prod_fn.L, prod_fn.K, prod_fn.v
            interior = (v * price / (lcost * cl ** v)) ** (1 / (1 - v))
            opt_labor = min(interior, cl * capital / ck)
            output = (opt_labor / cl) ** v
        if 0.01 <= opt_labor <= 1e6:
            break
    # Review: distractors are mistakes (the default ones were nearly always below the key). Labor: the
    # exponent 1/a instead of 1/(1-a) (Cobb-Douglas), the output price left out, the wage and price swapped in the
    # first-order condition, the capital-constrained / unconstrained bundle, the output at the optimum. Profit: the
    # revenue alone, the profit at those wrong labor amounts, the labor cost alone.
    def prod(Lx):
        if family == 'Cobb_Douglas':
            return Lx ** a * capital ** b
        return min(Lx / cl, capital / ck) ** v
    if family == 'Cobb_Douglas':
        wrong_l = [(a * price * capital ** b / lcost) ** (1 / a), (a * capital ** b / lcost) ** (1 / (1 - a)),
                   (a * lcost * capital ** b / price) ** (1 / (1 - a))]
    else:
        wrong_l = [interior if opt_labor < interior else cl * capital / ck,
                   (v / (lcost * cl ** v)) ** (1 / (1 - v)), (v * lcost / (price * cl ** v)) ** (1 / (1 - v))]
    wrong_l = [x for x in wrong_l if np.isfinite(x) and 0 < x < 1e7]
    if metadata['type'] == 'compute_labor':
        answer = round(float(opt_labor), 2)
        mistakes = wrong_l + [output]
    else:
        answer = round(float(price * output - lcost * opt_labor), 2)
        mistakes = [price * output, lcost * opt_labor] + [price * prod(x) - lcost * x for x in wrong_l]
    res = mistake_options(answer, mistakes)
    if res is None or answer < 0.05:
        return gen_profit_maximization(metadata)  # a key of a few cents leaves no distinct distractors: redraw
    answer_mask, options = res
    return {
        "capital": capital,
        "p_func": prod_fn.function_string,
        "cost": lcost,
        "price": price,
        "options_list": options,
        "correct_answer": answer_mask
    }

@function_registry.register('cost_minimization_problem')
def gen_cost_minimization_cobbdouglasprod_linearcost(metadata):
    return gen_cost_minimization(metadata, family = 'Cobb_Douglas')

@function_registry.register('cost_minimization_problem_linearprod_linearcost')
def gen_cost_minimization_linearprod_linearcost(metadata):
    return gen_cost_minimization(metadata, family = 'linear')

@function_registry.register('cost_minimization_problem_leonteifprod_linearcost')
def gen_cost_minimization_linearprod_linearcost(metadata):
    return gen_cost_minimization(metadata, family = 'Leonteif')

def gen_cost_minimization(metadata, family = 'Cobb_Douglas'):
    # Assumption: linear cost function (i.e., C = p_K * K + p_L * L)
    # Exogeneous: Production Function (f(.) = ... = Y), Min Output Level (Y), Output Price (P_Y), 
    #             Labor Cost (lcost), Capital Cost (kcost);
    # Optimizing: Capital (K) and Labor (L), subject to some f(K, L) >= Y. 

    #NOTE: possible extension is adding multiple cost functions
    no_answer =True
    while no_answer:
        prod_fn = ProductionFunction(family) 
        Y = round(np.random.rand() * 50 + 10, 2) # (Y) min output
        lcost = round(np.random.rand() * 10 + 0.5, 2)  # labor cost per unit (bounded away from 0)
        kcost = round(np.random.rand() * 10 + 0.5, 2)  # capital cost per unit

        if family == 'Cobb_Douglas':
            # Closed form (Narun's default; SLSQP often failed or stopped
            # early): with Y = L^a K^b, cost wL + rK, the tangency gives
            # K/L = b w / (a r) and L = (Y / (K/L)^b)^(1/(a+b)).
            a, b = prod_fn.L, prod_fn.K
            if a < 0.05 or b < 0.05:
                continue
            ratio = (b * lcost) / (a * kcost)
            labor = (Y / ratio ** b) ** (1 / (a + b))
            opt_inputs = np.array([labor, ratio * labor])
            if not np.all(np.isfinite(opt_inputs)) or opt_inputs.max() > 1e7:
                continue
            no_answer = False
            continue

        # Solve L*, K* numerically for other families; reject failures
        opt_inputs = prod_fn.find_cost_min(Y, lcost, kcost)
        if opt_inputs is not None and abs(prod_fn.production_output(*opt_inputs) - Y) <= 1e-4 * Y:
            no_answer = False  

    if metadata['type'] == 'compute_labor':
        # answer = Optimal labor
        answer = np.round(opt_inputs[0], 2)  
    elif metadata['type'] == 'compute_capital':
        # answer = Optimal capital
        answer = np.round(opt_inputs[1], 2) 
    else:
        # Minimum Cost
        answer = np.round(prod_fn.linear_cost(opt_inputs[0], opt_inputs[1], lcost, kcost), 2)

    # Review: the default distractors were nearly always below the key ("pick the largest" solved 13 of 15).
    # Cobb-Douglas distractors are the same quantity from a mistaken bundle: the price ratio inverted in the
    # tangency condition, the exponents swapped, equal exponents assumed (K/L = w/r), the 1/(a+b) power left out;
    # and the other input (labor <-> capital) or, for the cost, the labor or capital cost alone.
    mistakes = []
    if family == 'Cobb_Douglas':
        def bundle(ratio, power=True):
            lab = (Y / ratio ** b) ** (1 / (a + b)) if power else Y / ratio ** b
            return lab, ratio * lab
        bundles = [bundle(b * kcost / (a * lcost)), bundle(a * lcost / (b * kcost)), bundle(lcost / kcost),
                   bundle(b * lcost / (a * kcost), power=False)]
        idx = {'compute_labor': 0, 'compute_capital': 1}.get(metadata['type'])
        if idx is None:
            mistakes = [lcost * x + kcost * y for x, y in bundles] + [lcost * opt_inputs[0], kcost * opt_inputs[1]]
        else:
            mistakes = [bnd[idx] for bnd in bundles] + [opt_inputs[1 - idx]]
    res = mistake_options(float(answer), mistakes)
    if res is None or answer < 0.05:
        return gen_cost_minimization(metadata, family)  # a key of a few cents: redraw
    answer_mask, options = res

    return {
            "output": Y,
            "p_func": prod_fn.function_string,
            "l_cost": lcost,
            "k_cost": kcost,
            "options_list": options,
            "correct_answer": answer_mask
    }


# shephards_lemma (and the untemplated shephards_lemma_linear): replaced by filler_tools/shephards_lemma_tools.py
# (code templates), 5 October 2026.


@function_registry.register('marginal_production')
def gen_marginal_product(metadata):
    family = 'Cobb_Douglas'
    prod_fn = sample_prod_function(family,max_coeff=2)
    derivative_l = f"{prod_fn['L']}L^{round(prod_fn['L'] - 1, 2)}K^{prod_fn['K']}"
    derivative_k = f"{prod_fn['K']}L^{prod_fn['L']}K^{round(prod_fn['K'] - 1, 2)}"
    if metadata['type'] == 'marginal_labor':
        answer = derivative_l
    else:
        answer = derivative_k
    options = [derivative_l,derivative_k,f"{prod_fn['K']}L^{prod_fn['L']}K^{prod_fn['K']}",f"{prod_fn['L']}L^{prod_fn['L']}K^{prod_fn['K']}"]
    answer_mask,options = gen_answer_and_options(answer,num_options=4,options=options)
    return {
        "p_func": prod_fn['function_string'],
        "options_list":options,
        "correct_answer": answer_mask
    }

@function_registry.register('diminishing_marginal_product')
def gen_diminishing_marginal_product(metadata):
    """
    Types: {linear, cobb_douglas, quadratic}_{capital_fixed, labor_fixed}.
    capital_fixed varies labor L; labor_fixed varies capital K.

    Only truly diminishing marginal products are generated for the non-linear
    families: the Cobb-Douglas exponent of the varied input is in (0, 1) and
    the quadratic has a negative square term, with MP positive over the range.
    For linear functions the MP is the (constant) coefficient of the varied
    input. All numbers are computed from the displayed (rounded) parameters.
    """
    vary_labor = 'capital_fixed' in metadata['type']
    var_term = 'L' if vary_labor else 'K'  # the templates name the inputs L and K (template_fixes._dmp_rewrite)
    if 'cobb_douglas' in metadata['type']:
        family = 'cobb_douglas'
    elif 'quadratic' in metadata['type']:
        family = 'quadratic'
    else:
        family = 'linear'

    def marginal_product(prod_fn, var, fixed):
        if family == 'cobb_douglas':
            a_var = prod_fn['L'] if vary_labor else prod_fn['K']
            a_fix = prod_fn['K'] if vary_labor else prod_fn['L']
            return a_var * var ** (a_var - 1) * fixed ** a_fix
        if family == 'quadratic':
            lin, sq, cross = prod_fn['L'] if vary_labor else prod_fn['K']
            return lin + 2 * sq * var + cross * fixed
        return prod_fn['L'] if vary_labor else prod_fn['K']

    while True:
        if family == 'cobb_douglas':
            prod_fn = sample_prod_function(family, max_coeff=2)
            a_var = prod_fn['L'] if vary_labor else prod_fn['K']
            if not 0 < a_var < 1:
                continue
        elif family == 'quadratic':
            prod_fn = sample_prod_function(family, max_coeff=10, concave=True)
        else:
            prod_fn = sample_prod_function(family, max_coeff=10)
            if (prod_fn['L'] if vary_labor else prod_fn['K']) <= 0:
                continue
        start = int(rng.integers(1, 50))
        end = start + int(rng.integers(5, 50))
        fixed = int(rng.integers(5, 1000))
        if family == 'quadratic':
            # Re-review: with the fixed input up to 1000 the negative square term made output negative
            # ("students helped" of -176,068); output must be positive on the whole interval. Along the varied
            # input the function is a concave parabola, so its minimum on [start, end] is at an endpoint.
            def output(v):
                Lv, Kv = (v, fixed) if vary_labor else (fixed, v)
                return (prod_fn['a'] * Lv + prod_fn['b'] * Kv + prod_fn['c'] * Lv ** 2 + prod_fn['d'] * Kv ** 2
                        + prod_fn['e'] * Lv * Kv)
            if min(output(start), output(end)) <= 0:
                continue
        mp_start = round(float(marginal_product(prod_fn, start, fixed)), 2)
        mp_end = round(float(marginal_product(prod_fn, end, fixed)), 2)
        if family == 'linear':
            break
        # strictly decreasing and positive after rounding
        if mp_end >= 0.01 and mp_start - mp_end >= 0.01:
            break

    up = round(mp_start + float(rng.uniform(0.1, 1)) * max(mp_start, 1.0), 2)
    increases = f"The marginal product of {var_term} increases from {mp_start} to {up}"
    constant = f"The marginal product of {var_term} remains constant at {mp_start}"
    if family == 'linear':
        down = round(mp_start * float(rng.uniform(0.1, 0.9)), 2)
        decreases = f"The marginal product of {var_term} decreases from {mp_start} to {down}"
        answer = constant
    else:
        decreases = f"The marginal product of {var_term} decreases from {mp_start} to {mp_end}"
        answer = decreases
    options = [increases, constant, decreases]
    answer_mask, options = gen_answer_and_options(answer, num_options=3, options=options)

    return {
        "p_func": prod_fn['function_string'],
        "start": start,
        "end": end,
        "fixed": fixed,
        "options_list": options,
        "correct_answer": answer_mask
    }

@function_registry.register('supply_elasticity')
def gen_supply_elasticity(metadata):
    # price elasticity of supply, e = (dQ/dP) P / Q, for Q(P) shown in the question.
    # Types: Linear_supply_given, Quadratic_supply_given (the committed
    # templates call the latter Cobb_douglas_supply_given; the sampler renames it).
    # Review: the linear distractors were b, a + b and a (the same structure every time, all above 1 while
    # the key of a linear supply with a positive intercept is below 1). Distractors are now mistakes in the formula,
    # three chosen at random; draws with an elasticity below 0.05 or too few distinct mistakes are redrawn.
    while True:
        price = round(np.random.rand()*10+0.01,2)
        if metadata['type'] == "Linear_supply_given":
            family = 'linear'
            sup_fn = sample_supply_function(family)
            a, b = sup_fn['a'], sup_fn['b']
            q = a*price + b
            answer = a*(price/q)
            # P/Q inverted (Q/(aP)), the intercept's share (b/Q), P left out (a/Q), the intercept left out of Q
            # (aP/b), the inverse slope (P/(aQ)), unit elastic
            mistakes = [q/(a*price), b/q, a/q, a*price/b, price/(a*q), 1.0]
        else:
            family = 'quadratic'
            sup_fn = sample_supply_function(family,max_coeff=1)
            a, b, c = sup_fn['a'], sup_fn['b'], sup_fn['c']
            q = a*price**2 + b*price + c
            answer = (2*a*price+b)*(price/q)
            # the factor 2 of the derivative left out, P left out, P/Q inverted, the derivative alone, unit elastic
            mistakes = [(a*price+b)*price/q, (2*a*price+b)/q, q/((2*a*price+b)*price), 2*a*price+b, 1.0]
        if not answer >= 0.05:
            continue
        d = pick_mistakes(answer, mistakes, ndigits=6, rel=0.04, fill=False)
        if d is not None:
            break
    options = [answer] + d
    answer_mask,options = gen_answer_and_options(answer,options=options,num_options=4)
    return {
        "s_func": sup_fn['function_string'],
        "price": price,
        "options_list":options,
        "correct_answer": answer_mask
    }

# elasticity_of_substitution: replaced by filler_tools/elasticity_of_substitution_tools.py, code templates, 7 October 2026.

'''
Computes the amrginal change in optimal labor as cost of labor changes
for cobb douglas production function
'''
@function_registry.register('comp_stat_cobb')
def gen_comp_stat_cobb(metadata):
    family = 'Cobb_Douglas'
    prod_fn = sample_prod_function(family,max_coeff=1)
    while not 0.05 <= prod_fn['L'] <= 0.95:  # exponent 0 or 1 divides by zero below
        prod_fn = sample_prod_function(family,max_coeff=1)
    capital = round(np.random.rand()*10+0.01,2)
    price = round(np.random.rand()*10+0.01,2)
    coeff = capital**prod_fn['K']*price
    exp = 1/(prod_fn['L']-1)
    const = exp*(1/(coeff*prod_fn['L']))**exp
    answer = f"{const}C^{exp-1}"
    options = [answer,f"{const*rng.random()}C^{exp-1}",f"{coeff*rng.random()}C^{exp-1}",f"{const*rng.random()}C^{exp+1}"]
    answer_mask,options = gen_answer_and_options(answer,options=options,num_options=4)
    return {
        "capital": capital,
        "p_func": prod_fn['function_string'],
        "price": price,
        "options_list":options,
        "correct_answer": answer_mask
    }

@function_registry.register('comp_stat_leonteif')
def gen_comp_stat_leon(metadata):
    family = 'Leonteif'
    prod_fn = ProductionFunction(family)
    price = round(np.random.rand()*10+0.01,2)
    coeff = (1/(0.5*price*(1/prod_fn.L)**0.5))**-2
    exp = -2
    const = exp*coeff
    answer = f"{const}C^{exp-1}"
    # distractor coefficients clearly apart from the answer (x0.2-0.7 or x1.4-2.5)
    far = lambda: float(rng.choice([rng.uniform(0.2, 0.7), rng.uniform(1.4, 2.5)]))
    options = [answer,f"{const*far()}C^{exp-1}",f"{const*far()}C^{exp+1}",f"{-const*far()}C^{exp-1}"]
    answer_mask,options = gen_answer_and_options(answer,options=options,num_options=4)
    return {
        "p_func": prod_fn.function_string,
        "price": price,
        "options_list":options,
        "correct_answer": answer_mask
    }
'''
Takes in a prod fn in terms of L and K and a value for K 
returns a function in terms of just L
'''
def prod_fn_to_shortrun(prod_function,K):
    return

'''
Takes in a prod fn in terms of L and K and a value for K 
returns the simplified function averaged over L (i.e. divided by L)
'''
def prod_fn_to_arp(prod_function,K):
    return

'''
Takes in a prod fn in terms of L and K and a value for K 
returns the simplified function and takes derivative w.r.t. L
'''
def prod_fn_to_mrp(prod_function,K):
    return

# @function_registry.register('finding_tfp')
# def finding_tfp(metadata):
#     # TODO: documentation here

#     no_answer =True
#     while no_answer:
#         family = 'Cobb_Douglas' # TODO: ALTER SO TAGS LETS YOU CHANGE
#         prod_fn = ProductionFunction(family) 
#         capital = round(np.random.rand() * 10, 2)  # Random L
#         labour = round(np.random.rand() * 10, 2)  # Random K
#         answs = prod_fn.set_and_find_TFP(labour, capital)
#         if answs[0] is not None and answs[1] is not None:
#             A = answs[0]
#             total_output = answs[1]
#             no_answer = False # TODO: alter this if set_and_find_TFP might return None (i.e., if it in the future uses scipy)
        
#     if metadata['type'] == 'compute_simple_tfp':
#         # answer = Optimal labor
#         answer = np.round(A, 2)  
#     else:
#         # TODO: change if we have other tags
#         answer = np.round(A, 2)  
        
#     answer_mask, options = gen_answer_and_options(answer, num_options=4)

#     return {"p_func": prod_fn.function_string,
#             "capital": capital,
#             "labour": labour,
#             "total_output": total_output,
#             "options_list": options,
#             "correct_answer": answer_mask
#     }

# finding_tfp: replaced by filler_tools/tfp_tools.py (code templates), 5 October 2026.


# factor_shares_equilibrium: replaced by filler_tools/factor_shares_tools.py, code templates, 7 October 2026.

# comparative_stats_supply_price_shock: replaced by filler_tools/supply_elasticity_statics_tools.py (element elastic_inelastic_supply), code templates, 7 October 2026.

# fixed_factor_aggregation: replaced by filler_tools/fixed_factor_tools.py, code templates, 7 October 2026.
# lump_sum_tax_efficiency: replaced by filler_tools/non_distortionary_taxes_tools.py (element non_distortionary_taxes), code templates, 7 October 2026.
@function_registry.register('labor_tax_distortion')
def gen_labor_tax_distortion(metadata):
    variant = metadata.get("type", "labor_change")

    A = round(np.random.uniform(6, 12), 2)
    w = round(np.random.uniform(1.0, 3.0), 2)
    tau = round(np.random.uniform(0.1, 0.5), 2)  

    l_no_tax = A / 2
    l_with_tax = A / (2 + tau)

    c_no_tax = w * l_no_tax
    c_with_tax = (1 - tau) * w * l_with_tax

    leisure_no_tax = A - l_no_tax
    leisure_with_tax = A - l_with_tax

    u_no_tax = round(np.log(c_no_tax) + np.log(leisure_no_tax), 4)
    u_with_tax = round(np.log(c_with_tax) + np.log(leisure_with_tax), 4)

    utility_string = f"u(c, l) = ln(c) + ln({A} - l)"

    if variant == "labor_change":
        correct = round(l_no_tax - l_with_tax, 2)
        label = "reduction in labor supply caused by the labor income tax"

    elif variant == "consumption_drop":
        correct = round(c_no_tax - c_with_tax, 2)
        label = "reduction in consumption caused by the labor income tax"

    elif variant == "utility_loss":
        correct = round(u_no_tax - u_with_tax, 4)
        label = "utility loss caused by the labor income tax"

    else:
        raise ValueError(f"unsupported variant: {variant}")

    mask, opts = gen_answer_and_options(correct, 4)

    return {
        "utility_fn": utility_string,
        "w": w,
        "tau": tau,
        "A": A,
        "label": label,
        "options_list": opts,
        "correct_answer": mask,
    }

# TODO (CHICHI): move this to more appropriate file
@function_registry.register('duality_consumer_producer')
def gen_duality_linked_combined(metadata):
    variant = metadata.get("type", "type1")

    if variant == "type1":
        pf = ProductionFunction("Cobb_Douglas", max_coeff=2)
        theta = round(pf.K / (pf.K + pf.L), 2)
        correct = theta
        utility_fn = f"u(x, y) = x^{theta} * y^{1 - theta}"
        prod_fn = f"Y = K^{pf.K} * L^{pf.L}"
        label = "fraction of total cost spent on capital"

        mask, opts = gen_answer_and_options(correct, 4)
        return {
            "utility_fn": utility_fn,
            "theta": theta,
            "prod_fn": prod_fn,
            "label": label,
            "options_list": opts,
            "correct_answer": mask,
        }

    elif variant == "type2":
        theta = round(np.random.uniform(0.3, 0.7), 2)
        pf = ProductionFunction("Cobb_Douglas", max_coeff=2)
        pf.K = theta
        pf.L = round(1 - theta, 2)
        correct = round(1 - theta, 2)
        utility_fn = f"u(x_1, x_2) = x_1^{theta} * x_2^{1 - theta}"
        prod_fn = f"Y = K^{pf.K} * L^{pf.L}"
        label = "optimal fraction of total cost spent on labor"

        w = round(np.random.uniform(1.0, 4.0), 2)
        r = round(np.random.uniform(1.0, 4.0), 2)

        mask, opts = gen_answer_and_options(correct, 4)
        return {
            "utility_fn": utility_fn,
            "prod_fn": prod_fn,
            "theta": theta,
            "label": label,
            "w": w,
            "r": r,
            "options_list": opts,
            "correct_answer": mask,
        }

    elif variant == "type3":
        w = round(np.random.uniform(1.0, 4.0), 2)
        beta = round(np.random.uniform(0.4, 0.9), 2)
        Y = round(np.random.uniform(10, 100), 2)
        pf = ProductionFunction("Cobb_Douglas", max_coeff=2)
        pf.K = 0
        pf.L = beta
        correct = round(pf.find_labor_demand(Y=Y, kcost=1, lcost=w, capital_prime=1), 2)
        utility_fn = "u(c, l) = log(c) + log(1 - l)"
        prod_fn = f"Y = L^{pf.L}"
        label = "firm's labor demand"

        mask, opts = gen_answer_and_options(correct, 4)
        return {
            "utility_fn": utility_fn,
            "prod_fn": prod_fn,
            "Y": Y,
            "w": w,
            "label": label,
            "options_list": opts,
            "correct_answer": mask,
        }

    else:
        raise ValueError(f"Unsupported variant: {variant}")
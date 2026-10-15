# External packages
from scipy.optimize import minimize, Bounds, root_scalar
import numpy as np
# Local packages
from .micro_utility_utils import UtilityFunction
'''
Computes the profit-maximizing quantity for a given price and cost function.

'''
def gen_maximize_profit(metadata):
    p = np.random.rand()
    cost_function = UtilityFunction(family = 'quadratic').utility_func
    correct_answer = maximize_profit(p, cost_function)
    options = [correct_answer + np.random.rand() for _ in range(4)]
    return {
        'options_list': [options],
        'correct_answer': [[correct_answer == option for option in options]]
    }

def maximize_profit(p, cost_function, raise_errors = True):
    num_outputs = len(p)

    initial_guess = np.random.rand(num_outputs)
    profit_function = lambda q: p * q - cost_function(q)
    bounds = Bounds(lb=[0]*num_outputs, ub=[np.inf]*num_outputs, keep_feasible=[True]*num_outputs)
    neg_func = lambda q: -profit_function(q)
    result = minimize(neg_func, initial_guess, bounds = bounds)
    if result.success:
        return result.x
    else:
        if raise_errors:
            raise ValueError(result.message)
        else:
            return -np.inf
        
'''
Computes the profit-maximizing input choices for given input/output prices and a production function.

'''
def maximize_profit_with_inputs(p, w, r, production_function):
    initial_guess = np.random.rand(2)
    profit_function = lambda inputs: p * production_function(inputs) - w * inputs[0] - r * inputs[1]
    bounds = Bounds(lb=[0, 0], ub=[np.inf, np.inf], keep_feasible=[True, True])
    neg_func = lambda inputs: -profit_function(inputs)
    result = minimize(neg_func, initial_guess, bounds = bounds)#, options = {"disp": True})
    if result.success:
        return result.x
    else:
        raise ValueError(result.message)

'''
Computes the price associated with a profit-maximizing quantity, 
given the quantity and the cost function.

'''
def inverse_supply_curve(q, cost_function):
    # find a price p where q maximizes profit
    # TODO: check if ValueError in maximize_profit interrupts root_scalar
    profit_maximizing_condition = lambda p: q - maximize_profit(p, cost_function, raise_errors = False)
    # TODO: determine why convergence errors occur
    result = root_scalar(profit_maximizing_condition, x0 = np.array([np.random.rand()]))
    if result.converged:
        if result.root < 0:
            raise ValueError(f"p = {result.root} < 0")
        else:
            return result.root
    else:
        raise ValueError(result.flag)

'''
Computes the profit-maximizing quantity after a change in prices, 
given the price variation, cost function (and derivative), and quantity before the price variation.

'''
def maximize_profit_with_change_in_prices(price_variation, cost_function, cost_function_derivative, quantity_before):
    # compute the supply curve q(p), then invert it and evaluate at quantity_before
    price_before = inverse_supply_curve(quantity_before, cost_function_derivative)
    price_after = price_before + price_variation
    quantity_after = maximize_profit(price_after, cost_function)
    return quantity_after

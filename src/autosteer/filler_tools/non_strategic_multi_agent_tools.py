from scipy.optimize import root_scalar

from .utilities.supply_utils import maximize_profit_with_change_in_prices

'''
Computes the aggregate profit-maximizing quantity after a change in prices, 
given the number of firms, the price variation, the cost function derivative, and the quantity produced per firm before the price variation.

'''
def aggregate_supply_with_change_in_prices(num_firms, price_variation, cost_function_derivative, quantity_before):
    return num_firms * maximize_profit_with_change_in_prices(price_variation, cost_function_derivative, quantity_before)

'''
Computes the competitive equilibrium price given supply and demand functions.

'''
def competitive_equilibrium_price(supply_function, demand_function):
    equilibrium_condition = lambda p: supply_function(p) - demand_function(p)
    result = root_scalar(equilibrium_condition)
    if result.converged:
        if result.root < 0:
            raise ValueError(f"p = {result.root} < 0")
        else:
            return result.root
    else:
        raise ValueError(result.flag)
    # if solution does not exist, there is no competitive equilibrium price possible

'''
Computes the competitive equilibrium consumer demand after a change in the supply function,
given the demand function, the supply function before the change, and the shift in the supply function.

'''
def competitive_equilibrium_consumer_demand_varying_supply(demand_function, supply_function_before, supply_function_shift):
    supply_function_after = lambda p: supply_function_before(p) + supply_function_shift
    equilibrium_condition = lambda p: supply_function_after(p) - demand_function(p)
    result = root_scalar(equilibrium_condition)
    if result.converged:
        if result.root < 0:
            raise ValueError(f"p = {result.root} < 0")
        else:
            return demand_function(result.root)
    else:
        raise ValueError(result.flag)

'''
Computes the consumer demand after a change in the supply function before prices shift to a competitive equilibrium,
given the demand function and the supply function before the change.

'''
def consumer_demand_varying_supply_before_price_shift(demand_function, supply_function_before):
    equilibrium_condition = lambda p: supply_function_before(p) - demand_function(p)
    result = root_scalar(equilibrium_condition)
    if result.converged:
        if result.root < 0:
            raise ValueError(f"p = {result.root} < 0")
        else:
            return demand_function(result.root)
    else:
        raise ValueError(result.flag)

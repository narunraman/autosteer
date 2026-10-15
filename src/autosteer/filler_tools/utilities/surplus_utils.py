
import numpy as np
from scipy.optimize import root_scalar



def price_to_quantity(curve,price,funcs='linear'):
    if funcs == 'linear':
        return (price - curve[1])/curve[0]
    elif funcs == 'quadratic':
        return (-curve[1] + np.sqrt(curve[1]**2 - 4*curve[0]*(curve[2]-price)))/(2*curve[0])
    return (price - curve[1])/curve[0]

def quantity_to_price(curve,quantity,funcs='linear'):
    if funcs == 'linear':
        return curve[0]*quantity + curve[1]
    elif funcs == 'quadratic':
        return curve[0]*(quantity**2) + curve[1]*quantity + curve[2]
    return quantity*curve[0] + curve[1]
'''
Computes  surplus given a supply/demand curve and a price
'''
def surplus(curve, price=None):
    # find the quantity supplied at price
    quantity = price_to_quantity(curve,price)
    # compute the area of the triangle defined by the price, quantity, and the origin
    return 0.5 * abs(price- curve[1])* quantity

def total_surplus(demand_curve, supply_curve):
    price = quantity_to_price(demand_curve,find_equilibrium(demand_curve, supply_curve))
    return (price,(surplus(demand_curve, price) + surplus(supply_curve, price)))

def find_equilibrium(demand_curve, supply_curve,funcs='linear'):
    # find the quantity where quantity demanded equals quantity supplied
    if funcs == 'linear':
        equilibrium_condition = lambda p: demand_curve[0]*(p)+demand_curve[1] - (supply_curve[0]*(p)+supply_curve[1])
        result = root_scalar(equilibrium_condition, x0 = np.array([np.random.rand()]))
        if result.converged:
            # a 1-element array (array x0); return the scalar so float() and
            # brentq brackets work under numpy 2.4
            return np.asarray(result.root).item()
        else:
            raise ValueError(result.flag)
    elif funcs == 'quadratic': 
        a = demand_curve[0] - supply_curve[0]
        b = demand_curve[1] - supply_curve[1]
        c = demand_curve[2] - supply_curve[2]
        roots = np.roots([a,b,c])
        return max(roots)
        # equilibrium_condition = lambda p: demand_curve[0]*(p**2)+demand_curve[1]*p+ demand_curve[2] - (supply_curve[0]*(p**2)+supply_curve[1]*p+supply_curve[2])
    
    
'''
Computes the price associated with a profit-maximizing quantity,
given the supply function and demand funtion.
'''
def find_monopoly_quant(demand_curve,supply_curve,eq_quant):
    # Define the marginal cost (MC) function
    MC = lambda q: supply_curve[0]*(q)+supply_curve[1]

    # Define the marginal revenue (MR) function
    MR = lambda q:  (2*demand_curve[0]*(q)+demand_curve[1])

    # Define a function that calculates the difference between MC and MR
    MC_MR_diff = lambda q: MR(q) - MC(q)
    # Use a root finding algorithm to find the price where MC equals MR
    result = root_scalar(MC_MR_diff, x0 = np.array([np.random.rand()]),bracket=[0,eq_quant])
    if result.converged:
        return np.asarray(result.root).item()
    else:
        raise ValueError(result.flag)
"""
Computes the deadweight loss associated with a monopoly market
given the demand and supply curves.
"""
def deadweight_loss(demand_curve, supply_curve):
    #First find equilibrium price and surplus
    price,eq_surplus = total_surplus(demand_curve, supply_curve)
    #Find equilibrium quantity
    eq_quant = price_to_quantity(demand_curve,price)
    #Find quantity then price that maximizes revenue assuming monopoly
    m_quant = find_monopoly_quant(demand_curve,supply_curve,eq_quant)
    m_price = quantity_to_price(demand_curve,m_quant)
    #Find suppliers marginal cost at monopoly quantity
    s_price = quantity_to_price(supply_curve,m_quant)
    #monopoly surplus
    m_surplus = surplus(demand_curve, m_price) + surplus(supply_curve, s_price)+(m_price-s_price)*m_quant
    #deadweight loss is the difference between the surplus in the competitive market and the monopoly market
    dw_loss = eq_surplus - m_surplus
    # compute the area of the triangle defined by the price, quantity, and the intersection of the supply and demand curves
    return m_price,dw_loss


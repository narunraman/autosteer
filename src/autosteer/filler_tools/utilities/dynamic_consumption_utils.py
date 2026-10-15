
import numpy as np
from scipy.optimize import minimize

from .filler_utils import rng


### Portfolio Optimization

def two_asset_portfolio_optimization():
    # Generate random values for returns, bid-ask prices, and standard deviations
    expected_returns = rng.uniform(0.05, 0.08, 2)  # Expected returns for each asset
    bid_ask_spreads = rng.uniform(0.5, 1.5, 2)  # Random bid-ask spreads for transaction costs
    std_devs = rng.uniform(0.1, 0.15, 2)  # Standard deviation of each asset
    correlation = rng.uniform(-0.5, 0.5)  # Correlation between assets

    # Correlation matrix
    correlation_matrix = np.array([[1, correlation], [correlation, 1]])
    covariance_matrix = np.outer(std_devs, std_devs) * correlation_matrix

    # Define the objective function (negative Sharpe Ratio)
    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns) - np.dot(weights, bid_ask_spreads)
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        return -portfolio_return / portfolio_std  # Minimize negative Sharpe Ratio

    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    bounds = [(0, 1) for _ in range(2)]  # No short selling

    # Optimize portfolio
    initial_weights = np.array([0.5, 0.5])
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

    # Return the optimal allocation
    optimal_weights = result.x
    return {
        "expected_returns": expected_returns,
        "bid_ask_spreads": bid_ask_spreads,
        "std_devs": std_devs,
        "correlation": correlation,
        "Optimal Weights": optimal_weights
    }



def three_asset_min_risk_target_return(target_return=0.06):

    def compute_portfolio_result():

        # Generate random values for expected returns, bid-ask spreads, and standard deviations
        expected_returns = rng.uniform(0.05, 0.07, 3)
        bid_ask_spreads = rng.uniform(0.5, 2, 3)
        std_devs = rng.uniform(0.08, 0.15, 3)

        # Generate a random correlation matrix
        correlation_matrix = rng.uniform(-0.5, 0.5, (3, 3))
        np.fill_diagonal(correlation_matrix, 1)
        covariance_matrix = np.outer(std_devs, std_devs) * correlation_matrix

        # Objective function: minimize portfolio variance
        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            return portfolio_variance

        # Constraints and bounds
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}  # Target return
        ]
        bounds = [(0, 1) for _ in range(3)]  # No short selling

        # Optimize portfolio
        initial_weights = np.array([1/3, 1/3, 1/3])
        result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
        return result, expected_returns, bid_ask_spreads, std_devs, correlation_matrix

    result, expected_returns, bid_ask_spreads, std_devs, correlation_matrix = compute_portfolio_result()
    while result.success == False and any(result.x == 1):
        result, expected_returns, bid_ask_spreads, std_devs, correlation_matrix = compute_portfolio_result()

    # Return the optimal allocation
    optimal_weights = result.x

    # Create options
    options = []
    for i in range(3):
        option = f"Asset {i+1}: {optimal_weights[i]*100:.2f}%"
        options.append(option)
    
    return {
        "expected_returns": expected_returns,
        "bid_ask_spreads": bid_ask_spreads,
        "std_devs": std_devs,
        "correlation_matrix": correlation_matrix,
        "Optimal Weights": optimal_weights
    }






### Intertemporal Consumption Optimization


# Utility function with Constant Relative Risk Aversion (CRRA)
def crra_utility(c, gamma):
    if gamma == 1:  # Logarithmic utility for gamma = 1
        return np.log(c)
    else:
        return c ** (1 - gamma) / (1 - gamma)

# Lifetime utility calculation with discounting
def discounted_lifetime_utility(consumption, discount_factor, gamma):
    periods = len(consumption)
    return sum((discount_factor ** t) * crra_utility(consumption[t], gamma) for t in range(periods))

# Constraint: Budget constraint across all periods with interest
def budget_constraint(consumption, initial_wealth, income, interest_rate):
    periods = len(consumption)
    wealth = initial_wealth
    for t in range(periods):
        wealth = wealth + income[t] - consumption[t]
        wealth *= (1 + interest_rate)  # Accumulate interest
    return wealth

# Objective function for minimization (negative utility)
def objective(consumption, discount_factor, gamma):
    return -discounted_lifetime_utility(consumption, discount_factor, gamma)

# Solver function
def solve_optimal_consumption(initial_wealth, income, discount_factor, interest_rate, gamma):
    periods = len(income)
    # Initial guess for consumption (equal allocation)
    initial_guess = np.full(periods, (initial_wealth + sum(income)) / periods)

    # Define budget constraint as a dictionary for scipy.optimize
    constraints = {'type': 'eq', 'fun': lambda c: budget_constraint(c, initial_wealth, income, interest_rate)}

    # Solve for the consumption path that maximizes lifetime utility
    result = minimize(
        objective, 
        initial_guess, 
        args=(discount_factor, gamma), 
        constraints=constraints, 
        method='SLSQP', 
        bounds=[(0, None)] * periods  # Consumption must be non-negative
    )

    if result.success:
        return result.x  # Optimal consumption path
    else:
        raise ValueError("Optimization failed.")

# Random parameter generator
def generate_random_parameters(period_range = (3, 5)):
    initial_wealth = rng.uniform(0, 100000)  # Between 0 and 100k
    periods = rng.integers(*period_range)  # Between 5 and 20 periods
    income = rng.uniform(10000, 30000, periods)  # Between 10k and 30k for each period
    discount_factor = rng.uniform(0.9, 1.0)  # Between 0.9 and 1.0
    interest_rate = rng.uniform(0.01, 0.05)  # Between 1% and 5%
    gamma = rng.uniform(1, 3)  # Risk aversion coefficient between 1 and 3

    return initial_wealth, income, discount_factor, interest_rate, gamma, periods


def generate_closed_form_intertemporal_values():
    # Random values for parameters
    W0 = rng.uniform(5000, 10000)  # Initial wealth
    Y1 = rng.uniform(15000, 30000)  # Year 1 income
    Y2 = rng.uniform(10000, 20000)  # Year 2 income
    discount_rate = rng.uniform(0.9, 0.98)  # Discount rate
    interest_rate = rng.uniform(0.02, 0.06)  # Interest rate
    gamma = rng.uniform(1.5, 3)  # Coefficient of relative risk aversion
    
    # Discount factor and utility parameters
    beta = 1 / (1 + discount_rate)
    
    # Right-hand side of the budget constraint
    total_resources = W0 + Y1 + Y2 / (1 + interest_rate)
    
    # Solve using Euler equation for two periods
    ratio = (beta * (1 + interest_rate)) ** (1 / gamma)  # Optimal C2/C1 ratio
    C1 = total_resources / (1 + ratio / (1 + interest_rate))  # Solve for C1
    C2 = ratio * C1  # Use the ratio to find C2
    
    options_list = [[
        f"C1 = {C1 * 0.9:.2f}, C2 = {C2 * 0.9:.2f}",
        f"C1 = {C1 * 1.05:.2f}, C2 = {C2 * 1.05:.2f}",
        f"C1 = {C1 * 1.1:.2f}, C2 = {C2 * 1.1:.2f}",
        f"C1 = {C1:.2f}, C2 = {C2:.2f}"
    ]]

    answer_mask = [
        [0, 0, 0, 1]
    ]

    # Create a permutation to shuffle the options and answers
    permutation = rng.permutation(4)
    options_list = [np.array(options_list[0])[permutation].tolist()]
    answer_mask = [np.array(answer_mask[0])[permutation].tolist()]

    return {
        'initial_wealth': W0,
        'income': f"{Y1:.2f} in Year 1 and {Y2:.2f} in Year 2",
        'income_year1': Y1,
        'income_year2': Y2,
        'discount_rate': discount_rate,
        'interest_rate': interest_rate,
        'gamma': gamma,
        'periods': 2,
        'correct_answer': answer_mask,
        'options_list': options_list
    }

    question = f"""
    A medical practitioner has an initial wealth of ${W0:.2f} and expects to earn ${Y1:.2f} in Year 1 
    and ${Y2:.2f} in Year 2. The practitioner aims to maximize their lifetime utility, which follows 
    a CRRA utility function given by U(C) = C^(1 - gamma) / (1 - gamma), where gamma = {gamma:.2f}.

    Discount rate: {discount_rate:.2f}
    Interest rate: {interest_rate:.2f}

    Calculate the optimal consumption in Year 1 (C1) and Year 2 (C2).

    Options:
    (A) C1 = {C1 * 0.9:.2f}, C2 = {C2 * 0.9:.2f}
    (B) C1 = {C1 * 1.05:.2f}, C2 = {C2 * 1.05:.2f}
    (C) C1 = {C1 * 1.1:.2f}, C2 = {C2 * 1.1:.2f}
    (D) C1 = {C1:.2f}, C2 = {C2:.2f}
    """

    print(question)
    print(f"Correct Answer: (D) C1 = {C1:.2f}, C2 = {C2:.2f}\n")


import numpy as np
from scipy.optimize import minimize

def compute_optimal_bundle(pi, prices, W, utility_function):
    """
    Computes the optimal ex-ante consumption bundle in a two-state world.

    Parameters:
    - pi: List of probabilities for each state [pi1, pi2]
    - prices: List of prices for each good in each state [[p_x1, p_y1], [p_x2, p_y2]]
    - W: Initial wealth
    - utility_function: Function that computes utility given consumption of goods

    Returns:
    - c: Optimal ex-ante consumption bundle [c_x, c_y]
    """
    # Objective function: Negative expected utility
    def objective(c):
        EU = 0
        for i in range(2):
            c_x = c[0] / prices[i][0]  # Quantity of x in state i
            c_y = c[1] / prices[i][1]  # Quantity of y in state i
            U = utility_function(c_x, c_y)
            EU += pi[i] * U
        return -EU  # Negative for minimization

    # Budget constraint
    def constraint(c):
        return W - c[0] - c[1]

    # Initial guess
    c0 = [W / 2, W / 2]

    # Bounds: Can't spend negative amounts
    bounds = [(0, W), (0, W)]

    # Constraints
    cons = {'type': 'eq', 'fun': constraint}

    result = minimize(objective, c0, bounds=bounds, constraints=cons)
    return result.x

# Example usage:


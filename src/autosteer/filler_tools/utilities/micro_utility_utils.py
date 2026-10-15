import random
from scipy.optimize import minimize, Bounds, LinearConstraint
import numpy as np
import sympy as sp

class UtilityFunction:
    """
    A class that generates utility functions with random coefficients of different families.

    Parameters:
    - num_inputs: The number of inputs to the utility function.
    - offset: The offset of the utility function. One of -1, 0, 1. If -1 the offset is negative, if 0 there is no offset, if 1 the offset is positive.
    - family: The family of the utility function. One of 'linear', 'cobb-douglas', 'leontief', 'log', 'exponential', 'quadratic'.

    Supported families:
    - Linear: f(x) = c1 * x1 + c2 * x2 + ... + cn * xn + c0
    - Cobb-Douglas: f(x) = x1^c1 * x2^c2 * ... * xn^cn
    - Leontief: f(x) = min(x1/c1, x2/c2, ..., xn/cn)
    - Log: f(x) = log(x1^c1 + x2^c2 + ... + xn^cn)
    - Exponential: f(x) = exp(x1^c1 + x2^c2 + ... + xn^cn)
    - Quadratic: f(x) = c1 * x1^2 + c2 * x2^2 + ... + cn * xn^2

    Example:
    ```
    utility = UtilityFunction(num_inputs=2, family='linear')
    ```
    """
    def __init__(self, num_inputs: int = None, offset: int = 0, family: str | dict = None):
        """
        Initializes the utility function.

        Parameters:
        - num_inputs: The number of inputs to the utility function.
        - offset: The offset of the utility function (i.e., the y-intercept in 2D). One of -1, 0, 1. If -1 the offset is negative, if 0 there is no offset, if 1 the offset is positive. 
        - family: The family of the utility function. One of 'linear', 'cobb-douglas', 'leontief', 'log', 'exponential', 'quadratic'.

        Returns:
        - None

        Raises:
        - ValueError: If the family is not one of the supported families.

        Example:
        ```
        utility = UtilityFunction(num_inputs=2, family='linear')
        ```
        """

        assert family is not None and isinstance(family, dict) or isinstance(family, str), "Family must be specified. Supported families: 'linear', 'cobb-douglas', 'leontief', 'log', 'exponential', 'quadratic'. If using a dictionary, it must have 'family' key and coefficients key with values for the family and coefficients respectively."
        
        self.offset = offset
        if isinstance(family, dict):
            self.family = family['family']
            try:
                self.coefficients = family['coefficients']
                self.num_inputs = len(self.coefficients)
                # assert len(self.coefficients) == self.num_inputs, "Number of coefficients must be equal to the number of inputs."
            except KeyError:
                print("Coefficients not provided. Generating random coefficients.")
                self.coefficients = self._gen_random_coefficients()
        else:
            self.num_inputs = num_inputs
            self.family = family
            self.coefficients = self._gen_random_coefficients()
        self.utility_func = self._gen_utility()

    def _gen_random_coefficients(self):
        if self.family == 'linear':
            return np.random.rand(self.num_inputs)
        elif self.family == 'cobb-douglas':
            return np.round(np.random.rand(self.num_inputs),2)
        elif self.family == 'leontief':
            return np.random.rand(self.num_inputs)
        elif self.family == 'log':
            return np.random.rand(1)
        elif self.family == 'exponential':
            return np.random.rand(1)
        elif self.family == 'quadratic':
            return np.random.rand(self.num_inputs * 2)
        else:
            raise ValueError(f"Unsupported family: {self.family}")

    def _gen_utility(self):
        if self.family == 'linear':
            def utility_func(x):
                return np.dot(self.coefficients[:self.num_inputs], x) + self.coefficients[-1]
            return utility_func
        elif self.family == 'cobb-douglas':
            def utility_func(x):
                return np.prod([x[i] ** self.coefficients[i] for i in range(self.num_inputs)])
            return utility_func
        elif self.family == 'leontief':
            def utility_func(x):
                return np.min([x[i] / self.coefficients[i] for i in range(self.num_inputs)])
            return utility_func
        elif self.family == 'log':
            def utility_func(x):
                return np.log(np.sum([x[i] ** self.coefficients[i] for i in range(self.num_inputs)]))
            return utility_func
        elif self.family == 'exponential':
            def utility_func(x):
                return np.exp(np.sum([x[i] ** self.coefficients[i] for i in range(self.num_inputs)]))
            return utility_func
        elif self.family == 'quadratic':
            def utility_func(x):
                return np.sum([self.coefficients[i] * x[i] ** 2 for i in range(self.num_inputs)])
            return utility_func

    def find_optimal_point(self, p = None, budget = np.inf, initial_guess: float = None, raise_errors = True):
        """
        Finds the optimal point that minimizes the given function.

        Parameters:
        - p: The prices of the inputs to the utility function.
        - budget: The available budget for spending on the inputs.
        - initial_guess: The initial guess for the optimization algorithm. If None, a random guess is used.
        - raise_errors: If True, an error is raised on solver failure. If False, -np.inf is returned on solver failure instead.

        Returns:
        - The optimal point that minimizes the function, or -np.inf if the solver fails and raise_errors is False.

        Raises:
        - ValueError: If the optimization fails.
        """
        if initial_guess is None:
            initial_guess = np.random.rand(self.num_inputs)
        if p is None:
            p = np.array([0]*self.num_inputs)
        neg_func = lambda x: -self.utility_func(x)
        bounds = Bounds(lb=[0]*self.num_inputs, ub=[np.inf]*self.num_inputs, keep_feasible=[True]*self.num_inputs)
        constraint = LinearConstraint(np.array([p]), ub = budget, keep_feasible=True)
        result = minimize(neg_func, initial_guess, bounds = bounds, constraints = constraint)
        if result.success:
            return result.x
        else:
            if raise_errors:
                raise ValueError(result.message)
            else:
                return -np.inf
            
    def get_partial_derivative(self, input_num: int):
        """
        Returns the partial derivative of the utility function with respect to the given input.

        Returns:
        - The derivative of the utility function.
        """
        if self.family == 'linear':
            return self.coefficients[input_num]
        elif self.family == 'cobb-douglas':
            return f"{self.coefficients[input_num]}x^{self.coefficients[input_num] - 1}"
        elif self.family == 'leontief':
            return "Undefined"
        elif self.family == 'log':
            return f"{self.coefficients[0]}x^{self.coefficients[0] - 1}"
        elif self.family == 'exponential':
            return f"{self.coefficients[0]}x^{self.coefficients[0] - 1}"
        elif self.family == 'quadratic':
            return f"{2 * self.coefficients[input_num]}x{input_num + 1}"
        else:
            raise ValueError(f"Unsupported family: {self.family}")
    
    def __str__(self):
        # print the utility function as a mathematical function
        variables = ['x', 'y', 'z', 'w', 'v', 'u']
        if self.family == 'linear':
            return 'f(x) = ' + ' + '.join([f'{self.coefficients[i]} * x{i+1}' for i in range(self.num_inputs)]) + f' + {self.coefficients[-1]}'
        elif self.family == 'cobb-douglas':
            # TODO: MAKE SURE THIS CHANGE (ADDTION OF 'ELSE') DID NOT BREAK ANYTHING
            if self.num_inputs == 2:
                return f"U = x^{self.coefficients[0]} * y^{self.coefficients[1]}"
            else:
                terms = [f"x{i+1}^{self.coefficients[i]}" for i in range(len(self.coefficients))]
                return "U = " + " * ".join(terms)
        elif self.family == 'leontief':
            return 'f(x) = min(' + ', '.join([f'x{i+1}/{self.coefficients[i]}' for i in range(self.num_inputs)]) + ')'
        elif self.family == 'log':
            return 'U = ' + ' + '.join([f'{self.coefficients[i]}ln({variables[i]})' for i in range(self.num_inputs)]) + ')'
        elif self.family == 'exponential':
            return 'f(x) = exp(' + ' + '.join([f'x{i+1}^{self.coefficients[i]}' for i in range(self.num_inputs)]) + ')'
        elif self.family == 'quadratic':
            return 'f(x) = ' + ' + '.join([f'{self.coefficients[i]} * x{i+1}^2' for i in range(self.num_inputs)])


class DemandFunction:
    """
    A class to generate demand functions based on a given utility function and a budget.

    Parameters:
    - type: The type of the demand function: 'linear' or 'quadratic'.
    """

    def __init__(self, kind: str):
        assert kind in ['linear', 'quadratic', 'cobb-douglas'], "Currently, only Linear and Quadratic demand functions are supported."
        self.kind = kind
        # Generate random coefficients for the demand function
        if kind == 'linear':
            self.coefficients = [random.uniform(-10, -1), random.uniform(1, 10)]  # Ensure the slope is negative
        else:  # Quadratic
            self.coefficients = [random.uniform(-10, -1), random.uniform(-10, -1), random.uniform(1, 10)]  # Ensure the quadratic term is negative

    @classmethod
    def from_utility_fn(cls, utility_function: UtilityFunction, budget: float):
        """
        Initializes the demand function.

        Parameters:
        - utility_function: An instance of the UtilityFunction class.
        - budget: The available budget for spending on the inputs.
        """
        assert utility_function.family in ['cobb-douglas', 'linear'], "Currently, only Cobb-Douglas and Linear utility functions are supported."
        # Create a new instance of the class
        instance = cls(utility_function.family)
        # Set the utility function and budget of the instance
        instance.utility_function = utility_function
        instance.budget = budget
        return instance

    def _cobb_douglas_utility(self, prices):
        """
        Computes the demand for goods given Cobb-Douglas utility function and prices.

        Parameters:
        - prices: A list of prices for each good.

        Returns:
        - demand: A list of quantities demanded for each good.
        """
        alpha = self.utility_function.coefficients
        total_alpha = np.sum(alpha)
        demand = [(self.budget / prices[i]) * (alpha[i] / total_alpha) for i in range(self.utility_function.num_inputs)]
        return demand

    def _linear_utility(self, prices):
        """
        Computes the demand for goods given Linear utility function and prices.

        Parameters:
        - prices: A list of prices for each good.

        Returns:
        - demand: A list of quantities demanded for each good.
        """
        c = self.utility_function.coefficients
        marginal_utilities_per_dollar = [c[i] / prices[i] for i in range(self.utility_function.num_inputs)]
        best_good = np.argmax(marginal_utilities_per_dollar)
        demand = [0] * self.utility_function.num_inputs
        demand[best_good] = self.budget / prices[best_good]
        return demand
        

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
            elif self.utility_function.family == 'linear':
                return self._linear_utility(prices)
        else:
            if self.kind == 'linear':
                return [self.coefficients[0] * price + self.coefficients[1] for price in prices]
            elif self.kind == 'quadratic':
                return [self.coefficients[0] * price**2 + self.coefficients[1] * price + self.coefficients[2] for price in prices]

    # NOTE: not sure this is right
    def show_engel_curve(self):
        if self.utility_function.family == 'cobb-douglas':
            prices = sp.symbols(f'p1:{self.utility_function.num_inputs + 1}')
            budget = sp.Symbol('B')
            alpha = self.utility_function.coefficients
            total_alpha = np.sum(alpha)
            income_share = [alpha[i] * prices[i] / budget for i in range(self.utility_function.num_inputs)]
            return income_share

    def show_demand_function(self,budget):
        """
        Returns the functional form of the demand function.
        
        Returns:
        - demand_function: A sympy expression representing the demand function.
        """
        # Check that utility function is a class attribute
        assert hasattr(self, 'utility_function'), "Must call from_utility_fn to derive the demand function."
        assert self.utility_function.family in ['cobb-douglas', 'linear'], "Currently, only Cobb-Douglas and Linear utility functions are supported."
        prices = sp.symbols(f'p1:{self.utility_function.num_inputs + 1}')
        if self.utility_function.family == 'cobb-douglas':
            alpha = self.utility_function.coefficients
            total_alpha = np.sum(alpha)
            demand = [(budget / prices[i]) * (alpha[i] / total_alpha) for i in range(self.utility_function.num_inputs)]
            return demand
        elif self.utility_function.family == 'linear':
            c = self.utility_function.coefficients
            marginal_utilities_per_dollar = [c[i] / prices[i] for i in range(self.utility_function.num_inputs)]
            best_good = np.argmax(marginal_utilities_per_dollar)
            demand = [0] * self.utility_function.num_inputs
            demand[best_good] = budget / prices[best_good]
            return demand

    


    def __str__(self):
        if self.kind == 'linear':
            return f"Linear Demand Function: {self.coefficients[0]:.2f}p + {self.coefficients[1]:.2f}"
        else:
            return f"Quadratic Demand Function: {self.coefficients[0]:.2f}p^2 + {self.coefficients[1]:.2f}p + {self.coefficients[2]:.2f}"
        

def sample_utility_function(family='linear',max_coeff=2):
    prod_func = {}
    if family.lower() == 'linear':
        prod_func['x'] = round(np.random.rand()*10,2)
        prod_func['y'] = round(np.random.rand()*10,2)
        prod_func['function_string'] = f"{prod_func['x']}x + {prod_func['y']}y"
    elif family.lower() == 'cobb_douglas':
        prod_func['x'] = round(np.random.rand()*max_coeff,2)
        prod_func['y'] = round(np.random.rand()*max_coeff+0.01,2)
        prod_func['function_string'] = f"x^{prod_func['x']}y^{prod_func['y']}"
    elif family.lower() == 'quadratic':
        prod_func['a'] = round(np.random.rand()*max_coeff,2)
        prod_func['b'] = round(np.random.rand()*max_coeff,2)
        prod_func['c'] = round(np.random.rand()*max_coeff,2)
        prod_func['d'] = round(np.random.rand()*max_coeff,2)
        prod_func['e'] = round(np.random.rand()*max_coeff,2)
        prod_func['x'] = [prod_func['a'],prod_func['c'],prod_func['e']]
        prod_func['y'] = [prod_func['b'],prod_func['d'], prod_func['e']]
        var_list = ['x', 'y', 'x^2', 'y^2', 'xy']
        function_string = []
        for i, key in enumerate(['a','b','c','d','e']):
            if prod_func[key] == 0:
                continue
            function_string.append(f"{prod_func[key]}{var_list[i]}")
        prod_func['function_string'] = " + ".join(function_string)
    return prod_func
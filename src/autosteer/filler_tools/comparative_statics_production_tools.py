from .fill_registry import FillRegister
from .utilities.production_utils import gen_prod_decrease, gen_prod_increase, gen_labor_decrease, gen_labor_increase
function_registry = FillRegister()


@function_registry.register('tfp_shocks')
def gen_tfp_shocks_values(metadata):
    """
    Percent change in the competitive long-run price of output, which equals the unit cost of a constant-returns
    Cobb-Douglas firm Q = A L^a K^(1-a): P = (1/A) (w/a)^a (r/(1-a))^(1-a). Exact, not first-order:
      productivity (A) change of x%:  %dP = 1/(1 + x) - 1   (= -x/(1 + x))
      wage change of x%:              %dP = (1 + x)^a - 1   (a = labor share; exact for Cobb-Douglas)
    The capital-stock types (increase_capital / decrease_capital) were dropped (decision of 3 October 2026): with
    the price set by unit cost, a change in the capital stock does not move it, and the old answer had no basis.
    """
    gen = {'increase_productivity': gen_prod_increase, 'decrease_productivity': gen_prod_decrease,
           'increase_labor': gen_labor_increase, 'decrease_labor': gen_labor_decrease}.get(metadata['type'])
    if gen is None:
        raise ValueError("Invalid type for TFP shocks question")
    out = gen()
    # the "no change" distractor reads "0%", not "0.0%" (quality audit)
    out['options_list'] = [['0%' if str(o) in ('0.0%', '-0.0%', '0.00%', '-0.00%') else o for o in opts] for opts in out['options_list']]
    return out


import numpy as np
# from random import shuffle
from .utilities.production_utils import sample_prod_function,_opt_K2
from .fill_registry import FillRegister
from .utilities.filler_utils import gen_answer_and_options, band_options
function_registry = FillRegister()
 
@function_registry.register('dynamic_profit_max')
def gen_two_stage_profit_maximization(metadata):
    family = 'dynamic'
    prod_fn = sample_prod_function(family,max_coeff=1)
    # capital and prices at least 0.5 (a draw rounding to 0 printed "initial capital K_1 = 0.0")
    capital_1 = np.round(0.5 + np.random.rand()*9.5,2)
    price_1 = np.round(0.5 + np.random.rand()*9.5,2)
    if metadata['type'] == 'known_price':
        price_2 =np.round(0.5 + np.random.rand()*9.5,2)
    elif metadata['type'] == 'uncertain_price':
        # Generate three random prices between 1 and 10
        prices =np.round(np.random.uniform(1, 10, 3),2)
        # Generate three random probabilities and normalize them so they sum to 1
        probabilities = np.random.rand(3)
        probabilities /= probabilities.sum()
        # round to the displayed 2 decimals (the last one absorbs the rest) so
        # the expected price is computed from what the question shows
        probabilities = np.round(probabilities, 2)
        probabilities[2] = round(1 - probabilities[0] - probabilities[1], 2)
        if probabilities.min() < 0.05:   # "price 7.72 with probability 0.0"
            return gen_two_stage_profit_maximization(metadata)
        price_2 = np.sum(prices * probabilities)
        prices_probabilities_string = ", ".join(f"price {price} with probability {probability}" for price, probability in zip(prices, probabilities))
    elif metadata['type'] == 'costly_inspection':
        prod_fn  = sample_prod_function('dynamic', max_coeff=1)
        discount = np.random.rand()

        # 3-point price distribution
        prices   = np.round(np.random.uniform(1, 10, 3), 2)
        probs    = np.random.rand(3); probs /= probs.sum()
        price_str= ", ".join(f"{p:.2f} (prob {q:.2f})" for p, q in zip(prices, probs))

        # inspection fee
        c_i        = round(np.random.uniform(0.5, 3.0), 2)

        # --- optimise ----------------------------------------------------------
        def opt_K2(P):
            K2, π = _opt_K2(capital_1, discount, prod_fn, price_1, P)
            return K2, π

        # state-contingent plan (pay fee)
        K2_states, p_states = zip(*(opt_K2(P) for P in prices))
        EV_pay = sum(q*π for q, π in zip(probs, p_states)) - c_i

        # unconditional plan (skip fee)
        K2_skip, π_skip = opt_K2(sum(probs*prices))
        EV_skip = π_skip

        # --- build options ------------------------------------------------------

        values = {
            "p1": prices[0], 
            "p2": prices[1], 
            "p3": prices[2],
            "K2_p1": round(K2_states[0],2),
            "K2_p2": round(K2_states[1],2),
            "K2_p3": round(K2_states[2],2)
        }
        # create a uniformly biased set of wrong contingent capitals (+20 %)
        wrong_K2_p1 = round(K2_states[0] * 1.20, 2)
        wrong_K2_p2 = round(K2_states[1] * 0.80, 2)     # mix it up
        wrong_K2_p3 = round(K2_states[2] * 1.20, 2)
        wrong_K2_skip = round(K2_skip*0.80, 2)

        option_texts = [
            # This is the correct answer if EV_pay > EV_skip
            "Pay the fee. If tommorow's price = {price_1} set K2 = {K2_p1}; if tommorow's price = {p2} set K2 = {K2_p2}; if tommorow's price = {p3} set K2 = {K2_p3}".format(
                price_1=values['p1'],
                p2=values['p2'],
                p3=values['p3'],
                K2_p1=values['K2_p1'],
                K2_p2=values['K2_p2'],
                K2_p3=values['K2_p3']
            ),
            # incorrect answer if EV_pay > EV_skip
            "Pay the fee. If p_1 = {price_1} set K2 = {wrong_K2_p1}; if tommorow's price = {p2} set K2 = {wrong_K2_p2}; if tommorow's price = {p3} set K2 = {wrong_K2_p3}".format(
                price_1=values['p1'],
                p2=values['p2'],
                p3=values['p3'],
                wrong_K2_p1=wrong_K2_p1,
                wrong_K2_p2=wrong_K2_p2,
                wrong_K2_p3=wrong_K2_p3
            ),
            "Skip the fee and set a single K2 = {K2_skip}".format(
                K2_skip=round(K2_skip, 2)
            ),
            "Skip the fee and set a single K2 = {wrong_K2_skip}".format(
                wrong_K2_skip=wrong_K2_skip
            )
        ]

        # correct = 'A' if EV_pay > EV_skip else 'C'
        correct = option_texts[0] if EV_pay > EV_skip else option_texts[2]
        answer_mask, options_list = gen_answer_and_options(correct, options=option_texts)

        # print(answer_mask, options_list)
        return {
            "p_func"        : prod_fn['function_string'],
            "capital_1"     : capital_1,
            "price_1"       : price_1,
            "price_distro"  : price_str,
            "cost_func": "(K_1-K_2)^2",
            "inspection_cost": c_i,
            "correct_answer": answer_mask,
            "options_list": options_list,
            "discount"      : discount,
        }
    
    # rounded to the displayed 2 decimals; bounded optimiser (K2 >= 0): the
    # unbounded minimize could fail and return None, crashing the filler
    # a discount factor of a plausible size (0.01 to 1 gave factors such as 0.1)
    discount = round(float(np.random.uniform(0.5, 0.99)), 2)
    opt_capital, _ = _opt_K2(capital_1, discount, prod_fn, price_1, price_2)
    answer = opt_capital-capital_1
    if abs(answer) < 0.05:   # the key would print as 0.0 or -0.0
        return gen_two_stage_profit_maximization(metadata)

    # Distractors outside the range the answer can take given the rounded
    # inputs (the discount factor and probabilities are shown to 2 dp: +-0.005
    # on beta and on each probability); redraw if that range is too wide.
    d_ep = 0.005 * float(np.sum(prices)) if metadata['type'] == 'uncertain_price' else 0.0
    corners = [_opt_K2(capital_1, max(discount + db, 0), prod_fn, price_1, price_2 + de)[0] - capital_1
               for db in (-0.005, 0.005) for de in (-d_ep, d_ep)]
    got = band_options(answer, min(corners + [answer]), max(corners + [answer]))
    if not got:
        return gen_two_stage_profit_maximization(metadata)
    answer_mask, options = got
    if any(float(o) == 0 for o in options[0]):   # a distractor printed as 0.0
        return gen_two_stage_profit_maximization(metadata)
    if metadata['type'] == 'known_price':
        return {
        "capital_1": capital_1,
        "p_func": prod_fn['function_string'],
        "price_1": price_1,
        "cost_func": "(K_1-K_2)^2",
        "price_2": price_2,
        "options_list":options,
        "correct_answer": answer_mask,
        "discount": discount
    }
    elif metadata['type'] == 'uncertain_price':
        return {
        "capital_1": capital_1,
        "p_func": prod_fn['function_string'],
        "price_1": price_1,
        "cost_func": "(K_1-K_2)^2",
        "price_distro": prices_probabilities_string,
        "options_list":options,
        "correct_answer": answer_mask,
        "discount": discount
    }
    else:
        raise ValueError(f"Unknown metadata type: {metadata['type']}")
    



# entry_exit_decisions: replaced by filler_tools/entry_exit_tools.py, code templates, 7 October 2026.






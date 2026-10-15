
import numpy as np

from .fill_registry import FillRegister
from .utilities.filler_utils import rng, gen_answer_and_options, band

function_registry = FillRegister()


@function_registry.register("optimal_asset_allocation")
def gen_opt_asset_values(metadata):
    if metadata['difficulty_level'] == 0:
        num_assets = 2
    elif metadata['difficulty_level'] == 1:
        num_assets = 3
    

    # Generate random returns for each asset
    returns = rng.uniform(0.05, 0.15, num_assets)
    # Generate random standard deviations for each asset
    std_devs = rng.uniform(5, 15, num_assets)
    # Generate random bids for each asset
    bids = rng.uniform(5, 300, num_assets)
    # Generate random asks for each asset
    asks = bids + rng.uniform(5, 10, num_assets)
    
    if metadata['type'] == 'percentage':
        # formula = (ask - bid) / bid * 100 
        percentage_spread = ((asks - bids) / bids) * 100
    
    elif metadata['type'] == 'expected_tc':
        #  Expected Return Net of Transaction Costs
        # formula = return - (ask - bid) / bid * 100
        expected_tc = returns - ((asks - bids) / bids) * 100
    
    elif metadata['type'] == 'sharp_ratio':
        # Sharpe Ratio
        # formula = (return - risk_free_rate - transaction costs) / std_dev
        risk_free_rate = rng.uniform(0.01, 0.05)
        sharp_ratio = (returns - risk_free_rate - ((asks - bids) / bids) * 100) / std_devs
    
    elif metadata['type'] == 'round_trip':
        # Round Trip Transaction Costs
        # formula = (ask - bid) / bid * 100
        round_trip = ((asks - bids) / bids) * 100
        # minimize round trip transaction costs
        min_round_trip = np.min(round_trip)



# optimal_portfolio_choice: replaced by filler_tools/portfolio_bid_ask_tools.py, code templates, 7 October 2026.

@function_registry.register('arbitrage')
def gen_arbitrage_values(metadata):
    if metadata['type'] == 'two_assets':
        question = """In City A, a smartphone sells for $500. In City B, the same smartphone sells for $550. The cost to transport a smartphone from City A to City B is $30 per unit. Ignoring other transaction costs, is there an arbitrage opportunity, and what is the profit per unit?
        A) No, there's no arbitrage opportunity.
        B) Yes, buy in City A and sell in City B for a profit of $20 per unit.
        C) Yes, buy in City B and sell in City A for a profit of $50 per unit.
        D) Yes, buy in City A and sell in City B for a profit of $50 per unit."""

        # Prices and the transport cost are drawn at the 2 decimals shown, and the profit is computed from them.
        # Without arbitrage, the two prices differ by less than the transport cost, so shipping does not pay in
        # either direction. Distractor profits differ from the profit and from each other by at least $1.
        price_a = round(float(rng.uniform(400, 600)), 2)
        cost = round(float(rng.uniform(20, 40)), 2)

        is_arbitrage = rng.integers(0, 3) == 0

        def other_profits(k, avoid):
            out = []
            while len(out) < k:
                prf = round(float(rng.uniform(5, 50)), 2)
                if all(abs(prf - a) >= 1 for a in avoid + out):
                    out.append(prf)
            return out

        if is_arbitrage:
            price_b = round(price_a + cost + float(rng.uniform(5, 50)), 2)
            profit = round(price_b - price_a - cost, 2)
            correct = f"Yes, for a profit of ${profit:.2f} per unit."
            options = ["No, there's no arbitrage opportunity.", correct] + [
                f"Yes, for a profit of ${prf:.2f} per unit." for prf in other_profits(2, [profit])]
            answer_mask, options = gen_answer_and_options(correct, options=options)
        else:
            price_b = round(price_a + float(rng.uniform(-(cost - 0.5), cost - 0.5)), 2)
            options = ["No, there's no arbitrage opportunity."] + [
                f"Yes, for a profit of ${prf:.2f} per unit." for prf in other_profits(3, [])]
            answer_mask, options = gen_answer_and_options("No, there's no arbitrage opportunity.", options=options)

        return {
            'price_a': price_a,
            'price_b': price_b,
            'cost': cost,
            'correct_answer': answer_mask,
            'options_list': options
        }



@function_registry.register("intertemporal_consumption_smoothing")
def gen_intertemporal_consumption_smoothing(metadata):
    """
    CRRA consumption smoothing, u(c) = c^(1-g)/(1-g), max sum_t beta^t u(c_t)
    subject to sum_t c_t/(1+r)^t = W + sum_t y_t/(1+r)^t (income arrives from
    period 1; the parameter drawn in 0.9-0.99 is the discount factor beta,
    a convention; the sampler rewrites "discount rate" to "discount
    factor" in the templates). Euler: c_{t+1}/c_t = (beta (1+r))^(1/g).
    Both types ('numerical', 'closed_form') use this closed form, computed
    from the displayed (rounded) values.
    """
    r2 = lambda v: round(float(v), 2)
    if metadata['type'] == 'closed_form':
        periods = 2
    else:
        periods = int(rng.integers(*((3, 5) if metadata['difficulty_level'] == 1 else (2, 4))))
    while True:
        W = r2(rng.uniform(5000, 100000))
        incomes = [r2(v) for v in rng.uniform(10000, 30000, periods)]
        beta = r2(rng.uniform(0.9, 0.99))
        r = r2(rng.uniform(0.01, 0.06))
        gamma = r2(rng.uniform(1.5, 3))

        def path_g(g):
            # path growing by the factor g a year that exhausts lifetime wealth
            pv = W + sum(y / (1 + r) ** t for t, y in enumerate(incomes))
            c1 = pv / sum((g / (1 + r)) ** t for t in range(periods))
            return [c1 * g ** t for t in range(periods)]

        def path(b):
            return path_g((b * (1 + r)) ** (1 / gamma))

        answer = [r2(c) for c in path(beta)]
        # Review: the old fourth option was the key times 1.1 (same shape, so "the smaller of the two
        # look-alike paths" was the key). Every option now exhausts lifetime wealth and differs only in the
        # growth factor: three of four Euler-equation mistakes.
        mistakes = [
            path_g(1.0),                                     # flat: ignores discounting and interest
            path(1 / (1 + beta)),                            # the discount factor read as a rate
            path_g(((1 + r) / beta) ** (1 / gamma)),         # beta and (1 + r) inverted
            path_g((beta * (1 + r)) ** gamma),               # exponent gamma instead of 1/gamma
        ]
        pick = [int(i) for i in rng.choice(len(mistakes), 3, replace=False)]
        cands = [answer] + [[r2(c) for c in mistakes[i]] for i in pick]
        # distinct by more than 0.5% in some component
        ok = all(max(abs(a - b) / max(abs(b), 1) for a, b in zip(x, y)) > 0.005
                 for i, x in enumerate(cands) for y in cands[i + 1:])
        if not ok:
            continue
        # every distractor must leave the range the path can take for inputs
        # that round to the shown values (otherwise two options are plausible)
        def path_of(Wv, rv, gv, bv, *ys):
            g = (bv * (1 + rv)) ** (1 / gv)
            pv = Wv + sum(y / (1 + rv) ** t for t, y in enumerate(ys))
            c1 = pv / sum((g / (1 + rv)) ** t for t in range(len(ys)))
            return [c1 * g ** t for t in range(len(ys))]
        b_ = band(path_of, [W, r, gamma, beta, *incomes])
        if b_ is None:
            continue
        lo, hi = b_
        slack = np.maximum(0.011, 0.005 * np.maximum(np.abs(lo), np.abs(hi)))
        if all(np.any((np.asarray(c) < lo - slack) | (np.asarray(c) > hi + slack)) for c in cands[1:]):
            break
    texts = [', '.join(str(v) for v in c) for c in cands]
    correct = texts[0]
    rng.shuffle(texts)
    income = ', '.join(str(v) for v in incomes)
    income = ' and'.join(income.rsplit(',', 1))
    return {
        "initial_wealth": W,
        "income": income,
        "discount_rate": beta,
        "interest_rate": r,
        "gamma": gamma,
        "periods": periods,
        "correct_answer": [[t == correct for t in texts]],
        "options_list": [texts]
    }


# price_of_risk: replaced by filler_tools/price_of_risk_tools.py, code templates (review, October 2026).


@function_registry.register("state_based_consumption")
def gen_state_based_consumption_values(metadata):
    """
    Ex-ante allocation of wealth W between dollar spending on x and y before
    the state (A or B, with state-dependent prices) is known; in state i the
    consumer gets s_x/p_ix and s_y/p_iy units.
      cobb_douglas U = x^a y^b: E[U] is maximised at s_x = a/(a+b) W,
        s_y = b/(a+b) W (prices and probabilities drop out).
      linear U = a x + b y: all wealth goes to the good with the larger
        a E[1/p_x] vs b E[1/p_y].
    Closed forms from the displayed (rounded) values. Options are dollar
    pairs "(s_x, s_y)". (Default adopted per audit; the old options were the
    first two characters of the answer string.)
    """
    r2 = lambda v: round(float(v), 2)
    while True:
        pA = r2(rng.uniform(0.1, 0.9))
        pB = r2(1 - pA)
        W = r2(rng.uniform(50000, 200000))
        if metadata['type'] == 'cobb_douglas':
            a, b = r2(rng.uniform(0.1, 0.9)), r2(rng.uniform(0.1, 0.9))
            base = rng.uniform(5, 20, 2)
            prices = [[r2(v) for v in base], [r2(v) for v in base + rng.uniform(1, 10, 2)]]
            if abs(a - b) < 0.05 or abs(a + b - 1) < 0.05:
                continue
            sx, sy = r2(a / (a + b) * W), r2(b / (a + b) * W)
            candidates = [(sx, sy), (sy, sx), (r2(W / 2), r2(W / 2)), (r2(a * W), r2(b * W))]
            utility = f"U(x, y) = x^{a}y^{b}"
        else:
            a, b = r2(rng.uniform(0.2, 20)), r2(rng.uniform(0.2, 20))
            # state B dearer for both goods, as every template's story says (review: the stories said prices
            # "rise to" lower numbers); same construction as the Cobb-Douglas type
            base = rng.uniform(1, 10, 2)
            prices = [[r2(v) for v in base], [r2(v) for v in base + rng.uniform(0.5, 8, 2)]]
            mx = a * (pA / prices[0][0] + pB / prices[1][0])
            my = b * (pA / prices[0][1] + pB / prices[1][1])
            if abs(mx - my) < 0.02 * max(mx, my):
                continue  # near-tie: corner not clear-cut
            k = r2(mx / (mx + my))
            best = (W, 0.0) if mx > my else (0.0, W)
            candidates = [best, best[::-1], (r2(W / 2), r2(W / 2)), (r2(k * W), r2(W - k * W))]
            utility = f"U(x, y) = {a}x + {b}y"
        texts = [f"({c[0]}, {c[1]})" for c in candidates]
        if len(set(texts)) == 4:
            break
    answer = texts[0]
    rng.shuffle(texts)
    return {
        'initial_wealth': W,
        'pA': pA,
        'pB': pB,
        'pA_x': prices[0][0],
        'pA_y': prices[0][1],
        'pB_x': prices[1][0],
        'pB_y': prices[1][1],
        'utility_function': utility,
        'correct_answer': [[t == answer for t in texts]],
        'options_list': [texts]
    }

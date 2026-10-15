
# External packages
import numpy as np
# Local packages
from .utilities.surplus_utils import surplus, total_surplus, deadweight_loss,quantity_to_price
from .utilities.filler_utils import rng, gen_answer_and_options, band, band_options
from .fill_registry import FillRegister
function_registry = FillRegister()


'''
Computes consumer surplus given a demand curve and a price
Takes in Metadata dictionary
'''
MAX_CONSUMER_SLOPE = 10
MAX_PRODUCER_SLOPE = 10
MAX_CONSUMER_INTERCEPT = 20
MAX_PRODUCER_INTERCEPT = 5

@function_registry.register('consumer_surplus')
def gen_consumer_surplus(metadata):
    #generate two random numbers one negative and one positive for demand curve
    dup_options = True
    while dup_options:
        #demand curve is specified by a negative slope and positive y intercept
        demand_curve = [round(float(v), 2) for v in rng.random(2) * np.array([-MAX_CONSUMER_SLOPE, MAX_CONSUMER_INTERCEPT])]
        if abs(demand_curve[0]) < 0.05:
            continue  # a (rounded) zero slope has no well-defined surplus
        #generate a random price
        price = round(float(rng.random()*demand_curve[1]), 2)
        if price < 0.05:
            continue  # a price that rounds to 0 reads as a placeholder ("is 0.0?")
        demand_curve_string = f"{demand_curve[0]}Q + {demand_curve[1]}"
        consumer_surplus = surplus(demand_curve, price)
        if consumer_surplus < 0.05:
            continue  # a price at (or rounding to) the intercept leaves no surplus to compute
        answer_mask,options = create_answer_variants(consumer_surplus)
        #check if options are unique
        if len(set(options)) == len(options):
            dup_options = False
    return {
        "d_function": demand_curve_string,
        "price": price,
        "options_list":[options],
        "correct_answer": [answer_mask]
    }

'''
Computes producer surplus given a demand curve and a price
'''
@function_registry.register('producer_surplus')
def gen_producer_surplus(metadata):
    dup_options = True
    while dup_options:
        supply_curve = [round(float(v), 2) for v in rng.random(2) * np.array([MAX_PRODUCER_SLOPE, MAX_PRODUCER_INTERCEPT])]
        if abs(supply_curve[0]) < 0.05:
            continue  # a (rounded) zero slope has no well-defined surplus
        #generate a random price
        price = round(float((rng.random()*2+1)*(supply_curve[1])), 2)
        supply_curve_string = f"{supply_curve[0]}Q + {supply_curve[1]}"
        producer_surplus = surplus(supply_curve, price)
        if producer_surplus < 0.05:
            continue  # a price at (or rounding to) the intercept leaves no surplus to compute
        answer_mask,options = create_answer_variants(producer_surplus)
        #check if options are unique
        if len(set(options)) == len(options):
            dup_options = False
    return {
        "s_curve": supply_curve_string,
        "price": price,
        "options_list":[options],
        "correct_answer": [answer_mask]
    }

'''
Generate total surplus questions given metadata
'''
@function_registry.register('efficient_surplus')
def gen_efficient_surplus(metadata):
    dup_options = True
    while dup_options:
        demand_curve = [round(float(v), 2) for v in rng.random(2) * np.array([-MAX_CONSUMER_SLOPE, MAX_CONSUMER_INTERCEPT])]
        if abs(demand_curve[0]) < 0.05:
            continue  # a (rounded) zero slope has no well-defined surplus
        demand_curve_string = f"{demand_curve[0]}Q + {demand_curve[1]}"
        #generate a supply curve which begins below the demand curve so they intersect
        supply_curve = [round(float(v), 2) for v in rng.random(2) * np.array([MAX_PRODUCER_SLOPE, demand_curve[1]])]
        if abs(supply_curve[0]) < 0.05:
            continue  # a (rounded) zero slope has no well-defined surplus
        if demand_curve[1] - supply_curve[1] < 0.5:
            continue  # the curves must cross at a positive quantity, with a surplus worth asking about
        supply_curve_string = f"{supply_curve[0]}Q + {supply_curve[1]}"
        price,result_surplus = total_surplus(demand_curve, supply_curve)
        answer_mask,options = create_answer_variants(result_surplus)
        #check if options are unique
        if len(set(options)) == len(options):
            dup_options = False
    if metadata["type"]=="intersect_given":
        return {
            "d_curve": demand_curve_string,
            "s_curve": supply_curve_string,
            "price": float(price),
            "options_list":[options],
            "correct_answer": [answer_mask]
        }
    else:
        return {
            "d_curve": demand_curve_string,
            "s_curve": supply_curve_string,
            "options_list":[options],
            "correct_answer": [answer_mask]
        }

@function_registry.register('deadweight_loss')
def gen_deadweight_loss(metadata):
    dup_options = True
    while dup_options:
        demand_curve = [round(float(v), 2) for v in rng.random(2) * np.array([-MAX_CONSUMER_SLOPE, MAX_CONSUMER_INTERCEPT])]
        if abs(demand_curve[0]) < 0.05:
            continue  # a (rounded) zero slope has no well-defined surplus
        demand_curve_string = f"{demand_curve[0]}Q + {demand_curve[1]}"
        #generate a supply curve which begins below the demand curve so they intersect
        supply_curve = [round(float(v), 2) for v in rng.random(2) * np.array([MAX_PRODUCER_SLOPE, demand_curve[1]])]
        if abs(supply_curve[0]) < 0.05:
            continue  # a (rounded) zero slope has no well-defined surplus
        supply_curve_string = f"{supply_curve[0]}Q + {supply_curve[1]}"
        m_price, _ = deadweight_loss(demand_curve, supply_curve)
        m_price = round(float(m_price), 2)
        given = metadata["type"] == "monopoly_price_given"

        # DWL triangle between the competitive and the monopoly quantity; with
        # the monopoly price given, Q_m is read off the demand curve at P_m.
        def dwl(a, b, c, d, pm=None):
            qe = (b - d) / (c - a)
            qm = (pm - b) / a if pm is not None else (b - d) / (c - 2 * a)
            pm = a * qm + b if pm is None else pm
            return 0.5 * (pm - (c * qm + d)) * (qe - qm)
        params = demand_curve + supply_curve + ([m_price] if given else [])
        dw_loss = dwl(*params)
        # distractors outside the range consistent with the rounded inputs;
        # redraw when that range is too wide (tiny DWLs are ambiguous as printed)
        b_ = band(dwl, params)
        got = b_ and dw_loss >= 0.5 and band_options(dw_loss, *b_)   # (was > 0.05: options 0.06 / 0.08 / 0.1)
        if not got:
            continue
        answer_mask, options = got[0][0], got[1][0]
        dup_options = False
    if metadata["type"]=="monopoly_price_given":
        return {
            "d_curve": demand_curve_string,
            "s_curve": supply_curve_string,
            "m_price": m_price,
            "options_list":[options],
            "correct_answer": [answer_mask]
        }
    else:
        return {
            "d_curve": demand_curve_string,
            "s_curve": supply_curve_string,
            "options_list":[options],
            "correct_answer": [answer_mask]
        }


# @function_registry.register('deadweight_loss_incorrect')
# def gen_deadweight_loss_incorrect(metadata):


@function_registry.register('find_eq_price')
def gen_find_eq_price(metadata):
    """
    Inverse demand and supply in Q: linear P = aQ + b, quadratic
    P = aQ^2 + bQ + c. Coefficients are rounded to the displayed 2 decimals
    before solving; draws whose equilibrium quantity or price is not
    strictly positive are rejected.
    """
    while True:
        if 'linear' in metadata["type"]:
            demand_curve = [round(float(v), 2) for v in rng.random(2) * np.array([-MAX_CONSUMER_SLOPE, MAX_CONSUMER_INTERCEPT])]
            if abs(demand_curve[0]) < 0.05:
                continue  # a (rounded) zero slope has no well-defined surplus
            supply_curve = [round(float(v), 2) for v in rng.random(2) * np.array([MAX_PRODUCER_SLOPE, MAX_PRODUCER_INTERCEPT])]
            if abs(supply_curve[0]) < 0.05:
                continue  # a (rounded) zero slope has no well-defined surplus
            if demand_curve[0] >= 0 or supply_curve[0] - demand_curve[0] == 0:
                continue
            demand_curve_string = f"{demand_curve[0]}Q + {demand_curve[1]}"
            supply_curve_string = f"{supply_curve[0]}Q + {supply_curve[1]}"
            quantity = (demand_curve[1] - supply_curve[1]) / (supply_curve[0] - demand_curve[0])
            price = quantity_to_price(demand_curve, quantity, funcs='linear')
        else:
            demand_curve = [round(float(v), 2) for v in rng.random(3) * np.array([-MAX_CONSUMER_SLOPE, -MAX_CONSUMER_INTERCEPT, MAX_CONSUMER_INTERCEPT])]
            if abs(demand_curve[0]) < 0.05:
                continue  # a (rounded) zero slope has no well-defined surplus
            supply_curve = [round(float(v), 2) for v in rng.random(3) * np.array([MAX_PRODUCER_SLOPE, MAX_PRODUCER_INTERCEPT, demand_curve[2]])]
            if abs(supply_curve[0]) < 0.05:
                continue  # a (rounded) zero slope has no well-defined surplus
            demand_curve_string = f"{demand_curve[0]}Q^2 + {demand_curve[1]}Q + {demand_curve[2]}"
            supply_curve_string = f"{supply_curve[0]}Q^2 + {supply_curve[1]}Q + {supply_curve[2]}"
            a, b, c = (demand_curve[i] - supply_curve[i] for i in range(3))
            roots = [r.real for r in np.roots([a, b, c]) if abs(r.imag) < 1e-12 and r.real > 0]
            if len(roots) != 1:
                continue
            quantity = float(roots[0])
            price = quantity_to_price(demand_curve, quantity, funcs='quadratic')
        if not (quantity > 0.01 and price > 0.01):
            continue
        if 'price' in metadata["type"]:
            got = eq_price_options(demand_curve, supply_curve, float(price))
        else:
            got = eq_quantity_options(demand_curve, supply_curve, float(quantity))
        if got:
            answer_mask, options = got
            break
    return {
            "d_func": demand_curve_string,
            "s_func": supply_curve_string,
            "options_list": options,
            "correct_answer": answer_mask
        }
def _poly(coefs, q):
    """Value of the curve (coefficients highest power first) at quantity q."""
    return float(np.polyval(coefs, q))


def eq_price_options(demand, supply, price, num_options=4):
    """
    Options for an equilibrium price (review: the old random distractors all lay below the supply curve's
    price at Q = 0, so only the key was a price at which anything is supplied). Every option lies strictly between
    the supply curve's and the demand curve's price at Q = 0, the range in which the curves can cross. Distractors
    are mistakes where they fall in that range (midpoint of the two intercepts; the slopes' sign slipped; for
    quadratics the squared terms dropped, priced on either curve; the quantity halved or doubled), topped up with
    prices drawn uniformly from the range. At least max(0.05, 2% of the key) apart. Returns None if the range is too
    narrow (the caller redraws).
    """
    lo, hi = supply[-1], demand[-1]
    gap = max(0.05, 0.02 * abs(price))
    key = round(price, 2)
    cands = [(lo + hi) / 2]
    if len(demand) == 2:
        (ad, bd), (as_, bs) = demand, supply
        for den in (as_ + ad, as_ - ad):
            for num in (bd - bs, bd + bs):
                if den:
                    q = num / den
                    cands += [_poly(demand, q), _poly(supply, q)]
    else:
        q_lin = (demand[2] - supply[2]) / (supply[1] - demand[1]) if supply[1] != demand[1] else None
        if q_lin is not None:
            cands += [_poly(demand, q_lin), _poly(supply, q_lin)]
    q_star = None
    for q in np.roots(np.subtract(demand, supply)):
        if abs(q.imag) < 1e-12 and q.real > 0:
            q_star = float(q.real)
    if q_star is not None:
        for k in (0.5, 2.0):
            cands += [_poly(demand, k * q_star), _poly(supply, k * q_star)]
    return _options_in_range(key, cands, lo, hi, gap, num_options)


def _choke_quantity(demand):
    """Quantity at which the (decreasing) inverse demand reaches a price of 0."""
    qs = [float(q.real) for q in np.roots(demand) if abs(q.imag) < 1e-12 and q.real > 0]
    return min(qs) if qs else None


def eq_quantity_options(demand, supply, quantity, num_options=4):
    """
    Options for an equilibrium quantity (re-review: random distractors could exceed the quantity at which the
    demand price reaches 0). Every option lies strictly between a fifth of the key and that choke quantity. Distractors are mistakes in
    that range (the slopes' sign slipped; for quadratics the squared terms dropped; the quantity where demand meets
    supply's intercept; half and double the key), topped up with uniform draws; at least max(0.03, 5% of the key)
    apart. Returns None when the range is too narrow (the caller redraws).
    """
    hi = _choke_quantity(demand)
    if hi is None:
        return None
    key = round(quantity, 2)
    gap = max(0.03, 0.05 * abs(quantity))
    cands = [0.5 * quantity, 2.0 * quantity]
    if len(demand) == 2:
        (ad, bd), (as_, bs) = demand, supply
        for den in (as_ + ad, as_ - ad, -ad):
            for num in (bd - bs, bd + bs):
                if den:
                    cands.append(num / den)
    else:
        if supply[1] != demand[1]:
            cands.append((demand[2] - supply[2]) / (supply[1] - demand[1]))
        cands += [float(q.real) for q in np.roots(np.subtract(demand, [0, 0, supply[2]])) if abs(q.imag) < 1e-12]
    # not below a fifth of the key (a distractor of a few hundredths reads as a placeholder)
    return _options_in_range(key, cands, 0.2 * quantity, hi, gap, num_options)


def _options_in_range(key, cands, lo, hi, gap, num_options):
    """The key plus num_options - 1 values from cands (then uniform draws) strictly inside (lo, hi), all gap apart."""
    cands = list(cands)
    rng.shuffle(cands)
    chosen = []

    def ok(v):
        v = round(float(v), 2)
        return lo < v < hi and all(abs(v - c) >= gap for c in [key] + chosen)
    for c in cands:
        if len(chosen) == num_options - 1:
            break
        if np.isfinite(c) and ok(c):
            chosen.append(round(float(c), 2))
    for _ in range(200):
        if len(chosen) == num_options - 1:
            break
        v = float(rng.uniform(lo, hi))
        if ok(v):
            chosen.append(round(v, 2))
    if len(chosen) < num_options - 1:
        return None
    opts = [key] + chosen
    rng.shuffle(opts)
    return [[1 if o == key else 0 for o in opts]], [[str(o) for o in opts]]


"""
Generates the correct answer and alternative options for a given surplus question
"""
def create_answer_variants(answer,num_options=4):
    """Answer plus distractors kept clearly apart (see gen_answer_and_options); flat lists."""
    answer_mask, options = gen_answer_and_options(float(answer), num_options=num_options)
    return answer_mask[0], [float(o) for o in options[0]]

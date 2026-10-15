# Built-in packages
from copy import deepcopy
# External packages
import numpy as np
from autosteer.filler_tools.rng import pyrng
# Local packages


# The numpy Generator fillers draw from: the one bound for the question being filled (autosteer.filler_tools.rng)
from autosteer.filler_tools.rng import rng  # noqa: E402


def gen_answer_and_options(answer, num_options=4, options=None):

    # If options are provided, shuffle them and create the answer mask
    if options:
        pyrng.shuffle(options)
        answer_mask = [1 if opt == answer else 0 for opt in options]
        return [answer_mask], [options]

    # Ensure 'answer' is a list of numbers
    if isinstance(answer, (tuple, list, np.ndarray)):
        answers = [float(np.round(a, 2)) if isinstance(a, float) else a for a in answer]
    else:
        answers = [float(np.round(answer, 2))] if isinstance(answer, float) else [answer]

    # Initialize options with correct answers
    options = [deepcopy(answers)]

    jitter = 0.01
    attempt = 0
    # Generate incorrect options
    while len(options) < num_options:
        num_answers = len(answers) if len(answers) > 1 else 1
        # Generate random float(s) near the correct answers
        rand_option = np.round(rng.random(num_answers) * answers + rng.integers(0, 2), 2) + attempt * jitter
        # Keep distractors clearly apart from every existing option: more than
        # max(0.015, 3%) away in at least one component, so that they stay
        # distinct after 2-decimal rounding and are not near-twins of the answer.
        # A distractor printed as 0.0 or a few cents (rand * answer rounding to 0, 0.01, ...) reads as a placeholder:
        # redrawn when below min(0.05, a tenth of the answer), unless the answer has a 0 component (quality audit).
        tiny = np.maximum(0.005, np.minimum(0.05, 0.1 * np.abs(np.asarray(answers, dtype=float))))
        zero = bool((np.abs(rand_option) < tiny).any()) and not any(a == 0 for a in answers)
        if not zero and not any([(np.abs(rand_option - np.asarray(opt)) <= np.maximum(0.015, 0.03 * np.abs(np.asarray(opt)))).all()
                    for opt in options]):
            attempt = 0
            rand_option = rand_option
            options.append(rand_option.tolist())
        else:
            attempt += 1
            # print(rand_option)
            # print(options)

    # Shuffle the options
    pyrng.shuffle(options)
    # Convert all options to strings
    options = [str(opt) if isinstance(opt, float) or isinstance(opt, int) else ', '.join([str(o) for o in opt]) for opt in options]
    answers = str(answers) if isinstance(answers, float) or isinstance(answers, int) else ', '.join([str(a) for a in answers])

    # Create the answer mask
    answer_mask = [1 if opt == answers else 0 for opt in options]

    return [answer_mask], [options]


def gen_rounded_number(rng, lower_bound, upper_bound, round_to=2):
    return round(rng.uniform(lower_bound, upper_bound), round_to)


# ---------------------------------------------------------------------------
# Rounding-robust options
# ---------------------------------------------------------------------------
# Questions show parameters rounded to 2 decimals. A reader (or an
# independent solver) can only pin the answer down to the range of values
# the answer takes over all parameter values that round to what is shown.
# band() computes that range; band_options() builds distractors that lie
# clearly outside it, and returns None when the range is too wide relative
# to the answer (the question would be ambiguous as printed -> redraw).

def _half_width(v) -> float:
    """Rounding half-width of a shown value: integers are exact, else +-0.005."""
    v = float(v)
    return 0.0 if v == round(v) else 0.005


def band(f, params, exact=()):
    """(lo, hi) arrays of f over the corners of the rounding box of params.

    f takes the params positionally and returns a number or a sequence.
    Indices in `exact` are treated as exact (e.g. integers shown as '3.0').
    """
    import itertools
    params = [float(p) for p in params]
    widths = [0.0 if i in exact else _half_width(p) for i, p in enumerate(params)]
    free = [i for i, w in enumerate(widths) if w > 0]
    vals = []
    for signs in itertools.product((-1, 1), repeat=len(free)):
        x = list(params)
        for i, s in zip(free, signs):
            x[i] += s * widths[i]
        try:
            vals.append(np.atleast_1d(np.asarray(f(*x), dtype=float)))
        except (ZeroDivisionError, ValueError, OverflowError, FloatingPointError):
            return None
    if not free:
        vals.append(np.atleast_1d(np.asarray(f(*params), dtype=float)))
    vals = np.array(vals)
    if not np.all(np.isfinite(vals)):
        return None
    return vals.min(axis=0), vals.max(axis=0)


def band_options(answer, lo, hi, num_options=4, max_rel_width=0.25, fmt=None, spread=(0.4, 2.5)):
    """
    Answer plus distractors that each have at least one component clearly
    outside [lo, hi] (margin max(0.011, 0.5%) beyond the solver's slack).
    Returns ([mask], [options]) like gen_answer_and_options, or None if the
    band is wider than max_rel_width of the answer (ambiguous: redraw).
    """
    ans = np.atleast_1d(np.asarray(answer, dtype=float))
    lo, hi = np.atleast_1d(lo), np.atleast_1d(hi)
    scale = np.maximum(np.abs(ans), 0.05)
    if np.any((hi - lo) > max_rel_width * scale):
        return None
    slack = np.maximum(0.011, 0.005 * np.maximum(np.abs(lo), np.abs(hi)))
    r = lambda v: [round(float(x), 2) for x in v]
    opts = [r(ans)]
    for _ in range(500):
        if len(opts) >= num_options:
            break
        k = rng.uniform(*spread)
        if abs(k - 1) < 0.15:
            continue
        cand = ans * k if rng.random() < 0.8 else ans + rng.choice([-1, 1]) * rng.uniform(0.3, 1.0) * scale
        cand = np.asarray(r(cand))
        tiny = np.maximum(0.005, np.minimum(0.05, 0.1 * np.abs(ans)))
        if (np.abs(cand) < tiny).any() and not (np.round(ans, 2) == 0).any():
            continue   # a distractor printed as 0.0 or a few cents (quality audit)
        outside = (cand < lo - slack) | (cand > hi + slack)
        apart = all(np.any(np.abs(cand - np.asarray(o)) > np.maximum(0.015, 0.03 * np.abs(np.asarray(o))))
                    for o in opts)
        if outside.any() and apart:
            opts.append(list(cand))
    if len(opts) < num_options:
        return None
    fmt = fmt or (lambda v: str(v[0]) if len(v) == 1 else ', '.join(str(x) for x in v))
    texts = [fmt(o) for o in opts]
    correct = texts[0]
    pyrng.shuffle(texts)
    return [[1 if t == correct else 0 for t in texts]], [texts]

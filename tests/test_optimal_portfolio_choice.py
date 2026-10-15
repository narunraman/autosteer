"""optimal_portfolio_choice: keys recomputed from the question text.

The risk-free rate, risk aversion A and each fund's expected return, standard deviation and round-trip spread are
parsed. The utility E[r] - (A/2) Var(r) of the portfolio with a share w in the fund (return net of the spread) is
maximised over 0 <= w <= 1 by scipy (bounded search). one_asset: the key is 100 w*. best_fund: the fund whose
optimal portfolio has the highest utility. Exactly one option matches and it is the marked one; numeric options at
least 10% apart; every type, domain and perspective occurs, and so do keys of 0% and 100%; sampling is reproducible.
"""
import collections
import re

from scipy.optimize import minimize_scalar

from autosteer.release.rows import build_rows
from autosteer.sampler import ElementSampler

E = 'optimal_portfolio_choice'
RF = re.compile(r'risk-free account paying ([\d.]+)%')
AV = re.compile(r'risk aversion A = (\d+),')
ONE = re.compile(r'expected value of ([\d.]+)% and a standard deviation of ([\d.]+)%\. .+? spread of ([\d.]+)% of')
FUND = re.compile(r'Fund ([A-D]): expected return ([\d.]+)%, standard deviation ([\d.]+)%, spread ([\d.]+)%')


def best(mu, sd, s, rf, A):
    def u(w):
        return w * (mu - s) / 100 + (1 - w) * rf / 100 - A / 2 * (w * sd / 100) ** 2
    res = minimize_scalar(lambda w: -u(w), bounds=(0, 1), method='bounded', options={'xatol': 1e-10})
    w = min((res.x, 0.0, 1.0), key=lambda z: -u(z))
    return w, u(w)


def test_optimal_portfolio_choice():
    s = ElementSampler()
    seen = collections.Counter()
    for seed in range(600):
        (row,) = build_rows(s.sample_question(E, 151000 + seed), 'test')
        text, opts, k = row['question_text'], row['options'], row['correct_index']
        rf, A = float(RF.search(text).group(1)), float(AV.search(text).group(1))
        if row['type'] == 'one_asset':
            mu, sd, sp = (float(z) for z in ONE.search(text).groups())
            want = 100 * best(mu, sd, sp, rf, A)[0]
            vals = [float(o.rstrip('%')) for o in opts]
            assert [j for j, v in enumerate(vals) if abs(v - want) <= 0.0052] == [k], (text, opts, want)
            for i in range(4):
                for j in range(i):
                    assert abs(vals[i] - vals[j]) >= max(0.1 * max(vals[i], vals[j]), 1) - 0.011
            seen['zero' if want < 1e-6 else 'full' if want > 100 - 1e-6 else 'interior'] += 1
        else:
            funds = FUND.findall(text)
            assert [f[0] for f in funds] == list('ABCD')
            utils = {f'Fund {n}': best(float(mu), float(sd), float(sp), rf, A)[1] for n, mu, sd, sp in funds}
            want = max(utils, key=utils.get)
            assert sorted(opts) == sorted(utils) and opts[k] == want, (text, opts, utils)
        seen[row['type']] += 1
        seen[row['domain']] += 1
        seen[row['perspective']] += 1
    assert all(seen[t] for t in ('one_asset', 'best_fund', 'zero', 'full', 'interior', 'first_person',
                                 'second_person', 'man', 'woman', 'anon')), seen


def test_seeded_reproducibility():
    a, b = ElementSampler(), ElementSampler()
    for seed in (0, 7, 123456789):
        q = a.sample_question(E, seed)
        assert b.sample_question(E, seed) == q
        assert a.sample_question(E, seed, template_id=q['template_id']) == q

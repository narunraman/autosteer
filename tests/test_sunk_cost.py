"""sunk_cost: the key is the option with the highest future net value, whatever has already been spent.

Recomputed from the question text alone. The story states the sunk amount and, for continuing and for switching,
"<Gerund> <rest> would cost/take [another] X and bring in / be worth / save Y"; stopping "would cost nothing more"
(net 0). Each option is matched to its sentence (the option is the sentence's subject with the gerund made
imperative). Checks: the marked option has the highest net value, ahead by at least 3 units of the scenario's
scale; the option a sunk-cost reasoner picks (adding the sunk amount to continuing) is a wrong option; options are
distinct; both "switch" and "stop" keys occur; all five perspectives occur, including the AI assistant in charge;
there are 20 scenarios.
"""
import collections
import re

from helpers import sample
from autosteer.filler_tools.sunk_cost_tools import SCENARIOS

AMT = r'(\$[\d,]+|\d+ hours?)'
SENT = re.compile(rf'(?:^|(?<=\. ))([A-Z][\w-]*) ([^.]*?) would (?:cost|take) (?:another )?{AMT} and (?:bring in|be worth|save) {AMT}\.')
STOP = re.compile(r'(?:^|(?<=\. ))([A-Z][\w-]*) ([^.]*?) would (?:cost nothing more|take no more time) and (?:bring in|be worth|save) nothing\.')
SUNK = re.compile(rf'None of the {AMT} already spent can be recovered')


def val(s):
    return float(re.sub(r'[^\d]', '', s))


def test_sunk_cost():
    assert len(SCENARIOS) == 20
    keys, persp, scen = collections.Counter(), collections.Counter(), set()
    for s in sample('sunk_cost', 1200, seed=81):
        q = s.question
        sunk = val(SUNK.search(q).group(1))
        assert re.search(rf'already (?:spent|paid)(?: a non-refundable)? {re.escape(SUNK.search(q).group(1))}', q), q
        acts = SENT.findall(q)
        stop = STOP.findall(q)
        assert len(acts) == 2 and len(stop) == 1, q
        nets = {}
        for _ger, rest, cost, value in acts:
            nets[rest] = val(value) - val(cost)
        nets[stop[0][1]] = 0.0
        cont_rest = acts[0][1]  # the first stated course of action continues the original plan
        assert len(set(s.options)) == len(s.options) == 3
        by_option = []
        for o in s.options:
            rest = o.split(' ', 1)[1]
            assert rest in nets, (o, list(nets))
            by_option.append(nets[rest])
        best = max(range(3), key=lambda i: by_option[i])
        k = [k for k, sc in enumerate(SCENARIOS) if f"{sc['cont'][0]} {sc['cont'][2]}" in s.options]
        assert len(k) == 1, s.options
        step = SCENARIOS[k[0]]['step']  # the scenario's scale (amounts are multiples of it)
        assert all(by_option[best] >= v + 3 * step for i, v in enumerate(by_option) if i != best), (q, s.options)
        assert s.answer_index == best, (q, s.options, by_option)
        # the sunk-cost reasoner counts the sunk amount as lost unless it continues
        cont_best = s.options[best].split(' ', 1)[1] == cont_rest
        if not cont_best:  # a sunk-cost trap: counting the sunk amount makes continuing look best
            biased = [v + (sunk if o.split(' ', 1)[1] == cont_rest else 0) for o, v in zip(s.options, by_option)]
            trap = max(range(3), key=lambda i: biased[i])
            assert s.options[trap].split(' ', 1)[1] == cont_rest and trap != best, (q, s.options)
        keys['continue' if cont_best else 'stop' if by_option[best] == 0 else 'switch'] += 1
        if by_option[best] != 0:  # review round 2: the key trades off cost against value with the other course
            costs = {rest: val(cost) for _g, rest, cost, _v in acts}
            values = {rest: val(value) for _g, rest, _c, value in acts}
            kr = s.options[best].split(' ', 1)[1]
            other = next(r for r in costs if r != kr)
            assert (costs[kr] - costs[other]) * (values[kr] - values[other]) > 0, (q, s.options)
        if SCENARIOS[k[0]]['unit'] == 'time':  # savings are totals over a stated horizon (review)
            assert 'totals over the coming year; nothing is saved after that' in q, q
        persp[s.metadata['tags']] += 1
        if s.metadata['tags'] == 'ai_assistant':
            assert q.startswith('You are an AI assistant in charge of')
        scen.add(k[0])
    assert set(keys) == {'continue', 'stop', 'switch'}, keys
    assert min(keys.values()) >= 0.25 * sum(keys.values()), keys  # each keyed about a third of the time
    assert set(persp) == {'first_person', 'second_person', 'third_person;man', 'third_person;woman', 'ai_assistant'}
    assert len(scen) == 20

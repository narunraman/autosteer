"""Recover the Ellsberg (ambiguity_aversion) templates and label each one (October 2026).

Source: the v1 template pickle utility_gen/ambiguity.pkl (737 two-part GPT-4 templates with anonymous `{}` slots),
kept in the 2024 data-generation archive (not public). The fill code of the time
(rationality_benchmark fill_generations/ambiguity.ipynb, deleted in 655f635) filled the question slots with
(total, known, unknown) = (3n, n, 2n) and the option slot with a reward of 1, 2, 4 or 7.

For every template this script reads, from the texts alone:
  K   the colour with a known count (part 0's known-probability bet),
  U1  the unknown colour bet on in part 0, U2 the unknown colour added in part 1,
  the role of each question slot (total / known / unknown),
and keeps the template only if the structure is the Ellsberg one: part 0 = bet on K vs bet on U1; part 1 = bet on
U1 or U2 (the known-probability bet, 2/3) vs bet on K or U2. Options are stored with the known-probability bet
first in both parts. Templates whose bets cannot be read this way are dropped (the 165 "goods" templates have no
colours and no reward; others name an extra colour, lack a slot, or do not say which colours are unknown).

    python scripts/recover_ellsberg_templates.py <ambiguity.pkl> src/autosteer/filler_tools/ellsberg_templates.json
"""
from __future__ import annotations

import collections
import json
import re
import sys

import pandas as pd

COLOURS = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange', 'pink', 'silver', 'gold', 'golden',
           'brown', 'grey', 'gray', 'violet', 'turquoise', 'crimson', 'emerald', 'sapphire', 'ruby', 'azure', 'indigo',
           'amber']
COLOUR = re.compile(r'\b(' + '|'.join(COLOURS) + r')\b', re.I)
LOSING = re.compile(r'(\band 0\b|, 0\b|\b0 if\b|\b0 otherwise|\bnothing\b|\bno prize|\botherwise)', re.I)
COINS = re.compile(r'\b(gold|golden|silver)\s+(coin|coins|treasure|tokens?|pieces?)\b', re.I)


def bet_colours(option: str) -> set[str]:
    """Colours in the winning clause of a bet ("... a blue ball and 0 if you pick any other color" -> {blue})."""
    t = COINS.sub('', option)
    m = LOSING.search(t)
    if m and COLOUR.search(t[:m.start()]):
        t = t[:m.start()]
    return {c.lower() for c in COLOUR.findall(t)}


def label(q0: str, ops) -> tuple[dict | None, str]:
    if len(ops) != 2 or any(len(o) != 2 for o in ops):
        return None, 'shape'
    if q0.count('{}') not in (2, 3) or re.search(r'\{\w+\}', q0):
        return None, 'slots'
    if any(o.count('{}') != 1 for p in ops for o in p):
        return None, 'reward_slot'
    p0 = [bet_colours(o) for o in ops[0]]
    p1 = [bet_colours(o) for o in ops[1]]
    if not (all(len(s) == 1 for s in p0) and all(len(s) == 2 for s in p1)):
        return None, 'colours_in_options'
    shared = p1[0] & p1[1]
    if len(shared) != 1:
        return None, 'no_shared_colour'
    u2 = next(iter(shared))
    a, b = (next(iter(s)) for s in p0)
    if u2 in (a, b) or a == b:
        return None, 'structure'
    pair = r'[^.,;:]{0,25}\b(?:or|and)\b[^.,;:]{0,25}'
    unknown = [x for x in (a, b) if re.search(rf'\b{x}\b{pair}\b{u2}\b|\b{u2}\b{pair}\b{x}\b', q0, re.I)]
    if len(unknown) != 1:
        return None, 'unknown_pair'
    u1 = unknown[0]
    k = b if a == u1 else a
    if {frozenset(p1[0]), frozenset(p1[1])} != {frozenset({u1, u2}), frozenset({k, u2})}:
        return None, 'part1'
    # slot roles from the words after each slot
    roles = []
    for m in re.finditer(r'\{\}', q0):
        after = q0[m.end():m.end() + 60]
        words = after.split()[:3]
        if any(re.fullmatch(rf'{k}\w*[,.]?', w, re.I) for w in words):
            roles.append('known')
        elif re.search(rf'unknown|mix|either|\b{u1}\b|\b{u2}\b|not know|unclear|uncertain|could be|ratio', after, re.I):
            roles.append('unknown')
        else:
            roles.append('total')
    want = ['known', 'unknown'] if len(roles) == 2 else ['known', 'total', 'unknown']
    if sorted(roles) != sorted(want):
        return None, 'slot_roles'
    known0 = 0 if p0[0] == {k} else 1
    known1 = 0 if p1[0] == {u1, u2} else 1
    return {
        'q0': q0, 'roles': roles,
        'options0': [ops[0][known0], ops[0][1 - known0]],     # [bet on K, bet on U1]
        'options1': [ops[1][known1], ops[1][1 - known1]],     # [bet on U1 or U2, bet on K or U2]
        'known': k, 'unknown_bet': u1, 'added': u2,
    }, 'ok'


def main(src: str, out: str) -> None:
    d = pd.read_pickle(src)
    kept, stats = [], collections.Counter()
    for i, r in d.iterrows():
        lab, why = label(r.questions[0], r.options)
        stats[why] += 1
        if lab is None:
            continue
        q1 = re.sub(r'\byour can\b', 'you can', r.questions[1]).strip()
        kept.append({'source_row': int(i), 'domain': r.domain, **lab, 'q1': q1})
    # one template per distinct question text (the sampler identifies a template by its text; the first is kept)
    seen, unique = set(), []
    for t in kept:
        key = (t['q0'].strip(), t['q1'])
        if key not in seen:
            seen.add(key)
            unique.append(t)
    stats['duplicate'] = len(kept) - len(unique)
    doc = {'about': __doc__.strip().splitlines()[0], 'source': src, 'counts': dict(stats), 'templates': unique}
    with open(out, 'w') as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(dict(stats), len(unique), 'kept')


if __name__ == '__main__':
    main(*sys.argv[1:3])
